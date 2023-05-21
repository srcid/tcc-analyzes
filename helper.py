import numpy as np
import pandas as pd
from matplotlib import pyplot as plt

MILS_TO_SECS = 1e-3

solvers = 'CPLEX,GUROBI,SCIP,CBC'.split(',')
models = 'cb,cs'.split(',')

def plt_time_by_models(df: pd.DataFrame, f = None, fs = None):
    fig, ax = plt.subplots(figsize=fs)
    values = df.groupby(by='model').time.sum()

    for xtick, model_time in enumerate(values[models]):
        ax.bar(xtick, model_time)
        ax.text(xtick, model_time + values.max()/100, f'{model_time:.2f}', ha='center')

    ax.set_xticks(np.arange(2))
    ax.set_xticklabels(models)

    plt.ylim(top=values.max()*1.05)
    plt.ylabel('Tempo em segundos')
    plt.xlabel('Modelos')

    if f is not None:
        plt.savefig(f, bbox_inches='tight')

    plt.show()

def plt_time_by_solver(df: pd.DataFrame, f = None, fs = None):
    fig, ax = plt.subplots(figsize=fs)
    values = df.groupby('solver').time.sum()

    for xtick, time_s in enumerate(values[solvers]):
        ax.bar(x=xtick, height=time_s)
        ax.text(x=xtick, y=time_s + values.max()/100, s=f'{time_s:.2f}', ha='center')

    ax.set_xticks(np.arange(4))
    ax.set_xticklabels(solvers)

    plt.ylim(top=values.max()*1.05)
    plt.xlabel('Solvers')
    plt.ylabel('Tempo em segundos')

    if f is not None:
        plt.savefig(f, bbox_inches='tight')

    plt.show()

def plt_time_by_instance(df: pd.DataFrame, f = None, fs = None, fontsize = None, loc = None):
    fig, ax = plt.subplots(figsize=fs)
    width = 0.2
    instances = df['size'].unique()
    instances_idx = np.arange(len(instances))
    max = df['time'].max()

    cplex_bar = df.query('solver=="CPLEX"').sort_values('size')['time']
    gurobi_bar = df.query('solver=="GUROBI"').sort_values('size')['time']
    scip_bar = df.query('solver=="SCIP"').sort_values('size')['time']
    cbc_bar = df.query('solver=="CBC"').sort_values('size')['time']

    ax.bar(instances_idx-1.5*width, cplex_bar, width, label='CPLEX')
    ax.bar(instances_idx-0.5*width, gurobi_bar, width, label='GUROBI')
    ax.bar(instances_idx+0.5*width, scip_bar, width, label='SCIP')
    ax.bar(instances_idx+1.5*width, cbc_bar, width, label='CBC')

    for x,y in zip(instances_idx-1.5*width, cplex_bar):
        ax.text(x, y + max/100, f'{y:.2f}', ha='center', fontsize=fontsize)
    for x,y in zip(instances_idx-0.5*width, gurobi_bar):
        ax.text(x, y + max/100, f'{y:.2f}', ha='center', fontsize=fontsize)
    for x,y in zip(instances_idx+0.5*width, scip_bar):
        ax.text(x, y + max/100, f'{y:.2f}', ha='center', fontsize=fontsize)
    for x,y in zip(instances_idx+1.5*width, cbc_bar):
        ax.text(x, y + max/100, f'{y:.2f}', ha='center', fontsize=fontsize)

    ax.set_xticks(instances_idx)
    ax.set_xticklabels(map(lambda s: f'N={s}', instances))

    plt.ylim(top=max*1.1)
    plt.xlabel('Instâncias')
    plt.ylabel('Tempo em segundos')
    plt.legend(loc=loc)
    
    if f is not None:
        plt.savefig(f, bbox_inches='tight')

    plt.show()

def plt_model_solver(df: pd.DataFrame, f = None, fs = None):
    fig, ax = plt.subplots(figsize=fs)
    width=0.4
    instances = df['size'].unique()
    cb = df.query('model=="cb"').groupby('size').time.idxmin()
    cs = df.query('model=="cs"').groupby('size').time.idxmin()
    xtick = np.arange(10)
    max = df.loc[[*cb,*cs]]['time'].max()

    ax.bar(xtick-0.5*width, df.loc[cb].time, width, label='common blocks')
    ax.bar(xtick+0.5*width, df.loc[cs].time, width, label='common substring')

    for x, (_, row) in zip(xtick-0.5*width, df.loc[cb].iterrows()):
        y=row.time
        ax.text(x, y+max/100, f'{y:.2f}', ha='center')
        ax.text(x, y+max*0.06, row.solver, ha='center')
    for x, (_, row) in zip(xtick+0.5*width, df.loc[cs].iterrows()):
        y=row.time
        ax.text(x, y+max/100, f'{y:.2f}', ha='center')
        ax.text(x, y+max*0.06, row.solver, ha='center')

    ax.set_xticks(xtick)
    ax.set_xticklabels([ f'N={i}' for i in instances ])

    plt.ylim(top=max*1.15)
    plt.ylabel('Tempo em segundos')
    plt.xlabel('Instâncias')
    plt.legend(loc='upper left')
    
    if f is not None:
        plt.savefig(f, bbox_inches='tight')

    plt.show()