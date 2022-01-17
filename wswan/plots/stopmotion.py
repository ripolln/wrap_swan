#!/usr/bin/env python
# -*- coding: utf-8 -*-

import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec


def plot_segments_orientation(st_list, df_seg, st, xlength, ylength, N=1, batch=0):
    '''
    TODO: doc
    '''

    for j in range(N):
        i = j+ N*batch
        plt.subplot(2,N,j+1); plt.plot(st_list[i]['x'][:24*3], st_list[i]['y'][:24*3], 'r', linewidth=2)
        plt.subplot(2,N,j+1); plt.plot(st_list[i]['x'][24*3:], st_list[i]['y'][24*3:], c='limegreen', linewidth=3);
        plt.axis('square'); plt.xlim([-xlength/2, xlength/2]); plt.ylim([-ylength/2, ylength/2]);

    for j in range(N):
        i = j+ N*batch
        plt.subplot(2,N,j+1+N); plt.plot(df_seg['longitude'], df_seg['latitude'], '.'); plt.plot(st['lon'], st['lat'], '.-', c='silver');
        plt.plot([df_seg['longitude'].values[i], df_seg['longitude'].values[i+4]],
                 [df_seg['latitude'].values[i], df_seg['latitude'].values[i+4]], 'r', linewidth=2)
        plt.plot([df_seg['longitude'].values[i+4], df_seg['longitude'].values[i+5]],
                 [df_seg['latitude'].values[i+4], df_seg['latitude'].values[i+5]], c='limegreen', linewidth=3)
        plt.xlim([st.lon.min(),192]); plt.ylim([st.lat.min(),st.lat.max()])
        plt.axis('square')

    plt.gcf().set_size_inches(22*1.5, 8*1.5)

    return plt

def plot_grid_segments(st_list, df_seg, st, xlength, ylength, n_rows=5, n_cols=4, width=8, height=8):
    '''
    TODO: doc
    '''

    # number of steps of 1 day (24h*3 since dt=20')
    n_steps = 24*3

    # figure
    fig1 = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0, hspace=0)
    gr, gc = 0, 0

    for i, sti in enumerate(st_list):

        ax = plt.subplot(gs[gr, gc])

        ax.plot(sti['x'][:n_steps], sti['y'][:n_steps], 'r', linewidth=2)
        ax.plot(sti['x'][n_steps:], sti['y'][n_steps:], c='limegreen', linewidth=3); 
        ax.set_xticklabels([]); ax.set_yticklabels([]);
        ax.set_xlim([-xlength/2, xlength/2]); 
        ax.set_ylim([-ylength/2, ylength/2]);
        ax.text(-xlength/2+40000, -ylength/2+40000, i+1, fontsize=11, fontweight='bold')
        ax.patch.set_facecolor('blue')
        ax.patch.set_alpha(0.08)

        # counter
        gc += 1
        if gc >= n_cols:
            gc = 0
            gr += 1

    # figure
    fig2 = plt.figure(figsize=(width, height))
    gs = gridspec.GridSpec(n_rows, n_cols, wspace=0, hspace=0)
    gr, gc = 0, 0

    for i,sti in enumerate(st_list):

        ax = plt.subplot(gs[gr, gc])

        ax.plot(st['lon'], st['lat'], '-', c='silver');
        ax.plot([df_seg['longitude'].values[i], df_seg['longitude'].values[i+4]],
                 [df_seg['latitude'].values[i], df_seg['latitude'].values[i+4]], 'r', linewidth=2)
        ax.plot([df_seg['longitude'].values[i+4], df_seg['longitude'].values[i+5]],
                 [df_seg['latitude'].values[i+4], df_seg['latitude'].values[i+5]], c='limegreen', linewidth=3)
        ax.set_xticklabels([]); ax.set_yticklabels([]);
        ax.set_xlim([180,192]); ax.set_ylim([-20,-7.5])
        ax.text(180+0.35, -20+0.35, i+1, fontsize=11, fontweight='bold')
        ax.patch.set_facecolor('yellow')
        ax.patch.set_alpha(0.08)

        # counter
        gc += 1
        if gc >= n_cols:
            gc = 0
            gr += 1

    return fig1, fig2

