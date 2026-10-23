"""Integrity, leakage, architecture and end-to-end tests using temporary data."""

import json
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
import torch

from norway_energy_crisis.clean_data import (ROOT, START, END, REGIONS, STATIC, clean,
                                             normalize_weather, timestamps, validate_surveys)
from norway_energy_crisis.preprocess import PreparedData, calendar, encode_static, preprocess
from norway_energy_crisis import train_response as training


def fixture(root):
    raw=root/'raw';raw.mkdir()
    times=pd.date_range(START,END+' 23:00',freq='h')
    h=pd.DataFrame(dict(ID=[101,102,103],City=['Oslo','Bergen','Oslo'],Region=['Oslo','Bergen','Oslo'],
                        Price_area=['NO1','NO5','NO1'],Demand_data=['Yes']*3))
    h.to_csv(raw/'households.csv',index=False)
    survey=pd.read_csv(ROOT/'survey.csv');survey.to_csv(raw/'survey.csv',index=False)
    answers=pd.DataFrame({'ID':h.ID,'Q32':[1,1,3]})
    for q in STATIC:answers[q]=0 if q.startswith('Q27_') else 1
    answers.loc[1,'Q23']=np.nan
    answers.to_csv(raw/'answers.csv',index=False)
    rows=[]
    for i in h.ID:
        load=1+.2*np.sin(np.arange(len(times))*2*np.pi/24)+(i-101)*.1
        if i==101:load[0]=0
        rows.append(pd.DataFrame(dict(ID=i,Date=times.strftime('%Y-%m-%d'),Hour=times.hour+1,Demand_kWh=load)))
    pd.concat(rows).to_csv(raw/'demand.csv',index=False)
    rows=[];aggregate=[]
    for area in REGIONS:
        price=.5+.1*np.sin(np.arange(len(times))*.02)
        price[0]=-.01972
        price[times>='2022-01-01']+=2
        rows.append(pd.DataFrame(dict(Price_area=area,Date=times.strftime('%Y-%m-%d'),Hour=times.hour+1,
                                       Price_NOK_MWh=price*1000,Price_NOK_kWh=np.round(price,2))))
        aggregate.append(pd.DataFrame(dict(Price_area=area,Date=times.strftime('%Y-%m-%d'),Hour=times.hour+1,
                                            Demand_MWh=1.,Meter_count=1000,Demand_kWh_avg=1.)))
    pd.concat(rows).to_csv(raw/'prices.csv',index=False)
    pd.concat(aggregate).to_csv(raw/'demand_areas.csv',index=False)
    cache=root/'weather';cache.mkdir()
    utc=pd.date_range('2020-09-29','2022-04-02 23:00',freq='h')
    for area,(_,lat,lon) in REGIONS.items():
        request=dict(latitude=lat,longitude=lon,start_date='2020-09-29',end_date='2022-04-02',
                     hourly='temperature_2m',models='era5_land',timezone='UTC')
        payload=dict(hourly=dict(time=utc.strftime('%Y-%m-%dT%H:%M').tolist(),
                                temperature_2m=(10+5*np.sin(np.arange(len(utc))*.0005)).tolist()))
        (cache/f'{area}_era5_land.json').write_text(json.dumps(dict(request=request,response=payload)))
    cleaned=root/'cleaned';prepared=root/'prepared'
    clean(raw,cleaned,cache);preprocess(cleaned,prepared)
    return raw,cleaned,prepared


class PipelineTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        torch.set_num_threads(1)
        cls.temp=tempfile.TemporaryDirectory()
        cls.root=Path(cls.temp.name)
        cls.raw,cls.cleaned,cls.prepared=fixture(cls.root)
        cls.data=PreparedData(cls.prepared)

    @classmethod
    def tearDownClass(cls):cls.temp.cleanup()

    def test_cleaning_joins_cohort_and_units(self):
        audit=json.loads((self.cleaned/'manifest.json').read_text())
        self.assertEqual(audit['households'],3)
        self.assertEqual(audit['primary_households'],2)
        self.assertEqual(audit['genuine_zero_readings'],1)
        self.assertAlmostEqual(float(self.data.prices[0,0]),-.01972,places=6)
        self.assertEqual(float(np.load(self.cleaned/'demand.npy')[0,0]),0)
        self.assertEqual(self.data.preprocessor.ids,[101,102])
        with self.assertRaises(ValueError):timestamps(pd.DataFrame({'Date':['2021-01-01'],'Hour':[0]}))
        h=pd.read_csv(self.raw/'households.csv');a=pd.read_csv(self.raw/'answers.csv');s=pd.read_csv(self.raw/'survey.csv')
        a.loc[0,'Q32']=99
        with self.assertRaises(ValueError):validate_surveys(h,a,s)

    def test_duplicate_demand_and_region_join_fail(self):
        import shutil
        with tempfile.TemporaryDirectory() as folder:
            raw=Path(folder)/'raw';shutil.copytree(self.raw,raw)
            demand=raw/'demand.csv'
            with demand.open() as f:header=f.readline();row=f.readline()
            with demand.open('a') as f:f.write(row)
            with self.assertRaisesRegex(ValueError,'Duplicate household/hour'):
                clean(raw,Path(folder)/'bad',self.root/'weather')

    def test_dst_and_unexpected_weather_gaps(self):
        index=pd.date_range('2021-03-28','2021-03-28 23:00',freq='h')
        utc=pd.date_range('2021-03-27 23:00','2021-03-28 21:00',freq='h')
        payload={'hourly':{'time':utc.strftime('%Y-%m-%dT%H:%M').tolist(),'temperature_2m':list(range(len(utc)))}}
        values,changes=normalize_weather(payload,index)
        self.assertEqual(values[2],(values[1]+values[3])/2)
        self.assertEqual(len(changes['spring_hours_interpolated']),1)
        fall=pd.date_range('2021-10-31','2021-10-31 23:00',freq='h')
        utc=pd.date_range('2021-10-30 22:00','2021-10-31 22:00',freq='h')
        payload={'hourly':{'time':utc.strftime('%Y-%m-%dT%H:%M').tolist(),'temperature_2m':list(range(25))}}
        values,changes=normalize_weather(payload,fall)
        self.assertEqual(values[2],2)
        self.assertEqual(len(changes['autumn_second_hours_omitted']),1)
        payload['hourly']['temperature_2m'][5]=None
        with self.assertRaises(ValueError):normalize_weather(payload,fall)

    def test_training_only_scales_static_unknowns_and_origins(self):
        data=self.data;train=data.partitions==0
        self.assertAlmostEqual(data.state['price_mean'],float(data.prices[:,train].mean()),places=6)
        self.assertLess(data.state['price_mean'],float(data.prices.mean()))
        static,categories=encode_static(np.array([[1.],[np.nan]]),np.array([0,0]))
        self.assertEqual(static[1,-1],1)
        for i,split in enumerate(['train','validation','test']):
            for origin in data.origins[split]:
                self.assertTrue(np.all(data.partitions[origin-24:origin+24]==i))
        i=data.origins['validation'][0]
        a=data.batch(np.array([0]),np.array([i]),5)
        b=data.batch(np.array([0]),np.array([i]),24)
        torch.testing.assert_close(a['history'],b['history'][:,-5:])
        torch.testing.assert_close(a['target'],b['target'])
        np.testing.assert_allclose(calendar(np.array(['2021-01-01T00','2021-01-01T12'],dtype='datetime64[h]'))[:,0],[0,1])

    def test_architectures_gradients_checkpoint_and_inference(self):
        batch=self.data.batch(np.array([0,1]),np.repeat(self.data.origins['train'][0],2),24)
        for architecture in training.ARCHITECTURES:
            model=training.ResponseModel(architecture,24,self.data.static.shape[1]).eval()
            predicted=training.rollout(model,batch,3)
            self.assertTrue(torch.isfinite(predicted).all())
            self.assertTrue((predicted>=0).all())
            predicted.sum().backward()
            self.assertGreater(float(model.head[-1].weight.grad.abs().sum()),0)
            with tempfile.TemporaryDirectory() as folder:
                path=Path(folder)/'best.pt'
                training.atomic_torch_save(training.checkpoint(model,self.data,{},6,1.),path)
                restored,_=training.load_model(path)
                with torch.no_grad():torch.testing.assert_close(training.rollout(restored,batch,3),predicted)
                api=training.Predictor(path)
                origin=self.data.origins['train'][0]
                indices=np.arange(origin-24,origin)
                result=api.predict(self.data.demand[:,indices],self.data.prices[self.data.area_index[:,None],indices],
                                   self.data.temperature[self.data.area_index[:,None],indices],self.data.times[indices],
                                   self.data.preprocessor.ids,self.data.prices[self.data.area_index,origin])
                np.testing.assert_allclose(result,predicted[:,0].detach().numpy(),rtol=1e-5)

    def test_no_future_demand_or_temperature_and_price_free_invariance(self):
        batch=self.data.batch(np.array([0,1]),np.repeat(self.data.origins['train'][0],2),5)
        model=training.ResponseModel('GRU',5,self.data.static.shape[1],False).eval()
        torch.nn.init.normal_(model.head[-1].weight,std=.1)
        changed={k:v.clone() for k,v in batch.items()}
        changed['target'].fill_(1e6)
        changed['future_temperature']=torch.full((2,24),1e6)
        changed['prices'].fill_(1000)
        changed['history'][:,:,1].fill_(-1000)
        with torch.no_grad():torch.testing.assert_close(training.rollout(model,batch,6),training.rollout(model,changed,6))
        captured=[]
        hook=model.register_forward_pre_hook(lambda _,args:captured.append(args[0].detach().clone()))
        with torch.no_grad():training.rollout(model,batch,3)
        hook.remove()
        for h in captured[1:]:torch.testing.assert_close(h[:,-1,2],batch['history'][:,-1,2])

    def test_interrupted_run_resumes_identically(self):
        interrupted=self.root/'interrupted'
        original=training.rollout
        calls=0
        def interrupt(model,batch,horizon=24):
            nonlocal calls
            if model.training:
                calls+=1
                if calls==2:raise KeyboardInterrupt('Simulated interruption')
            return original(model,batch,horizon)
        with patch.object(training,'rollout',interrupt):
            with self.assertRaises(KeyboardInterrupt):
                training.train_run(self.data,interrupted,'GRU',5,42,True,6,1,4)
        saved=torch.load(interrupted/'last.pt',weights_only=True)
        self.assertEqual(saved['epoch'],1)
        training.train_run(self.data,interrupted,'GRU',5,42,True,6,1,4)
        reference=self.root/'uninterrupted'
        training.train_run(self.data,reference,'GRU',5,42,True,6,1,4)
        resumed,_=training.load_model(interrupted/'best.pt')
        expected,_=training.load_model(reference/'best.pt')
        for name,value in resumed.state_dict().items():
            torch.testing.assert_close(value,expected.state_dict()[name],rtol=0,atol=0)
        self.assertFalse((interrupted/'last.pt').exists())

    def test_complete_comparison_freezes_before_test_and_resumes(self):
        output=self.root/'runs'
        original=training.forecast
        def guarded(model,data,split,batch_size=128):
            if split=='test':self.assertTrue((output/'selection.json').is_file())
            return original(model,data,split,batch_size)
        with patch.object(training,'forecast',guarded):
            training.compare(self.prepared,output,epochs=6,batches=1,batch_size=4)
        self.assertEqual(json.loads((output/'status.json').read_text())['state'],'complete')
        selection=json.loads((output/'selection.json').read_text())
        self.assertEqual(len(selection['winner_runs']),3)
        self.assertEqual(len(selection['control_runs']),3)
        self.assertEqual(len(list(output.glob('*/best.pt'))),15)
        self.assertEqual(len(list(output.glob('*/last.pt'))),0)
        self.assertTrue((output/'REPORT.md').is_file())
        training.compare(self.prepared,output,epochs=6,batches=1,batch_size=4)


if __name__=='__main__':unittest.main()
