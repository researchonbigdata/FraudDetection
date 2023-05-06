# -*- coding: utf-8 -*-
"""
Created on Mon Apr 25 16:32:43 2022

@author: witch
"""


import numpy as np

import math
import csv

import gzip
import csv
import os
import io
import pandas as pd
import copy
from scipy import stats


def time_gap(vol_list):
    if vol_list.empty:
        return 0
    days = [g for n, g in vol_list.groupby(
        pd.Grouper(key='start_datetime', freq='D'))]
    gaps = []

    for day in days:
        day['pre_start_datetime'] = day['start_datetime'].shift(1)
        day['gap'] = day['start_datetime']-day['pre_start_datetime']
        day['gap'] = day['gap'].dt.total_seconds()
        gaps.append(day['gap'].mean())
    # gaps=pd.to_datetime(gaps)
    # gaps=gaps.dt.total_seconds()
    mean = np.mean(gaps)
    return mean


def time_gap2(voc_list):
    if not voc_list['start_datetime']:
        return 0
    pytime = []
    for time in voc_list['start_datetime']:
        pytime.append(time.to_pydatetime())
    pytime = np.sort(pytime)

    gaps = []
    for i in range(1, len(pytime)):
        if pytime[i].day != pytime[i-1].day:
            continue
        gaps.append((pytime[i]-pytime[i-1]).total_seconds())
    if len(gaps) == 0:
        return 0
    return np.mean(gaps)


def time_gap_static(voc_list):
    if not voc_list['start_datetime']:
        return []
    gaps = []
    pytime = []
    for time in voc_list['start_datetime']:
        pytime.append(time.to_pydatetime())
    pytime = np.sort(pytime)
    for i in range(1, len(pytime)):
        if pytime[i].day != pytime[i-1].day:
            continue
        gaps.append((pytime[i]-pytime[i-1]).total_seconds())
    return gaps
    pass


def time(vol_list):  # vol_list 是一个人一周的通话记录
    if vol_list.empty:
        return 0
    # split by day
    vol_list['start_day'] = vol_list['start_datetime'].dt.date
    vol_list['start_day'] = pd.to_datetime(vol_list['start_day'])
    vol_list['time_on_day'] = vol_list['start_datetime']-vol_list['start_day']
    vol_list['time_on_day'] = vol_list['time_on_day'].dt.total_seconds()
    days = [g for n, g in vol_list.groupby(
        pd.Grouper(key='start_datetime', freq='D'))]
    median = []
    for day in days:
        time_mean = day['time_on_day'].median()
        median.append(time_mean)
        pass
    median = np.mean(median)
    return median


def time2(voc_list: pd.DataFrame):
    time = []
    if not voc_list['hour']:
        return [-1 for i in range(24)]

    dis = [0 for i in range(24)]
    for hour in voc_list['hour'].tolist():
        dis[hour] += 1
    sum = np.sum(dis)
    for idx in len(dis):
        dis[idx] /= sum

    return dis


def connector_duplicate(vol_list_this_week, vol_list_last_week):
    if len(vol_list_last_week) == 0 or len(vol_list_this_week) == 0:
        return 0
    connect1 = set(vol_list_last_week['opposite_no_m'])
    connect2 = set(vol_list_this_week['opposite_no_m'])
    union_set = connect1 | connect2
    interset = connect1 & connect2
    return len(interset)/len(union_set)


def connector_duplicate2(voc_last_week, voc_this_week):
    if (not voc_last_week['opposite_no_m']) or (not voc_this_week['opposite_no_m']):
        return 0
    connect1 = set(voc_last_week['opposite_no_m'])
    connect2 = set(voc_this_week['opposite_no_m'])
    union_set = connect1 | connect2
    interset = connect1 & connect2
    if len(union_set) == 0:
        return 0
    return len(interset)/len(union_set)
    pass


def arpu(vol_list):
    pass


def area_change(vol_list):
    if vol_list.empty:
        return 0
    change_times = vol_list['county_name_y'].nunique()
    # area=None
    # change_times=0
    # for idx,data in vol_list.iterrows():
    #     city=str(data['city_name_y'])
    #     county=str(data['county_name_y'])
    #     city=' '.join([city,county])
    #     #print(city)
    #     if area!=city:
    #         change_times+=1
    #         area=city
    # change_times-=1
    # if change_times==-1:
    #     change_times=0
    return change_times


def area_change2(voc_list):
    if not voc_list['county_name_y']:
        return 0
    areas = set()
    for city, county in zip(voc_list['city_name_y'], voc_list['county_name_y']):
        areas.add(' '.join([city, county]))
    return len(areas)


def areas(voc_list):
    if not voc_list['county_name_y']:
        return []
    areas = list()
    for city, county in zip(voc_list['city_name_y'], voc_list['county_name_y']):
        areas.append(' '.join([city, county]))
    return areas


def recall_rate(vol_list):
    vol_list = pd.DataFrame(vol_list)
    if vol_list.empty:
        return 0
    n_call_to = 0.0
    n_call_back = 0.0
    for idx, row in vol_list.iterrows():
        if row['calltype_id'] == 1:
            n_call_to += 1
            calltime = row['start_datetime']
            call_back = vol_list.loc[(vol_list['start_datetime'] > row['start_datetime']) & (
                vol_list['calltype_id'] == 2) & (vol_list['opposite_no_m'] == row['opposite_no_m'])]
            if len(call_back) > 0:
                n_call_back += 1
    if n_call_to == 0:
        return 0
    return n_call_back/n_call_to


def recall_rate2(voc_list):
    if not voc_list['opposite_no_m']:
        return 0
    called = set()
    calltimes = 0
    callback_times = 0
    for p, calltype in zip(voc_list['opposite_no_m'], voc_list['calltype_id']):
        if calltype == 1:
            called.add(p)
            calltimes += 1
        elif calltype == 2:
            if p in called:
                called.remove(p)
                callback_times += 1

    if calltimes == 0:
        return 0
    return callback_times/calltimes


def energy_dispersion(vol_list):

    if vol_list.empty:
        return (0, 0)
    vol_list['counting'] = vol_list.groupby(
        'opposite_no_m')['opposite_no_m'].transform('count')  # 按照对方的电话分组
    vol_list['counting'] = vol_list['counting']/len(vol_list)
    each_one = list((vol_list.drop_duplicates(
        subset='opposite_no_m'))['counting'])
    mean = np.mean(each_one)
    varia = np.var(each_one)
    return (mean, varia)


def energy_dispersion2(voc_list):
    if not voc_list['opposite_no_m']:
        return 0
    p_count = {}
    count = 0
    for p in voc_list['opposite_no_m']:
        if p not in p_count:
            p_count[p] = 1
        else:
            p_count[p] += 1
        count += 1
    p_count = list(p_count.values())
    p_count = np.divide(p_count, count)
    return np.var(p_count)


def var2(voc_list):
    if voc_list['call_dur']:
        return np.var(voc_list['call_dur'])
    return 0


def mean_voc_time(voc_list):

    mean = 0.0
    if voc_list.empty:
        return 0
    mean = voc_list['call_dur'].mean()
    return mean


def mean_voc_time2(vol_list):
    if not vol_list['call_dur']:
        return 0
    return np.mean(vol_list['call_dur'])


def n_unique_persons2(voc_list):
    persons = set(voc_list['opposite_no_m'])
    return len(persons)


def trying():
    return 12


def normalize(mx: np.ndarray):  # 均值方差归一化
    shape = mx.shape
    mx = mx.reshape((-1, shape[-1]))
    for k in range(mx.shape[-1]):
        mx[:, k] = (mx[:, k]-np.mean(mx[:, k]))/np.std(mx[:, k])
    mx = mx.reshape(shape)
    return mx
