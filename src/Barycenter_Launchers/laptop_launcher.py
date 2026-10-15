#!/usr/bin/env python3
"""Mac launch menu for the recovered barycenter experiments (stdlib only)."""
import argparse
from datetime import datetime
import fcntl
import json
import os
from pathlib import Path
import shlex
import signal
import subprocess
import sys
import time

HERE = Path(__file__).resolve().parent
PACKAGE = HERE.parent / 'closing_gap_reproduction'
PYTHON = Path('/usr/bin/python3')
RUNS = HERE / 'results'
THREADS = 4


def choose(prompt, choices, default=1, interactive=True):
    if not interactive:
        return choices[default-1][0]
    print('\n'+prompt)
    for i, (_, label) in enumerate(choices, 1):
        print('  {}. {}'.format(i, label))
    while True:
        value = input('Choice [{}]: '.format(default)).strip() or str(default)
        if value.isdigit() and 1 <= int(value) <= len(choices):
            return choices[int(value)-1][0]
        print('Enter one of the numbers above.')


def select_values(prompt, allowed, interactive):
    if not interactive:
        return list(allowed)
    while True:
        s = input('{} [{}]: '.format(prompt, ','.join(allowed))).strip()
        values = s.upper().replace(',', ' ').split() if s else list(allowed)
        if values and len(values) == len(set(values)) and all(v in allowed for v in values):
            return values
        print('Choose one or more of: '+', '.join(allowed))


def settings(mode, interactive):
    opts = dict(mode=mode, digits=['0','1','2'], classes=['A','C','D'],
                rule='notebook-subsets', ts=['1','2'], minutes=60, ellipse='all')
    if mode == 'ellipse':
        opts['ellipse'] = choose('Ellipse experiment', [
            ('all','Full comparison: t=1, t=2, ten sampled runs, saved reference'),
            ('t1','Deterministic t=1 only'), ('t2','Deterministic t=2 only'),
            ('reference','Draw the saved reference and inputs only')], interactive=interactive)
    if mode in ('ellipse','all') and opts['ellipse'] != 'reference':
        opts['solver'] = choose('Ellipse LP solver', [
            ('mosek','MOSEK (recommended; sparse LP)'),
            ('gurobi','Gurobi')], default=1, interactive=interactive)
    if mode in ('mnist','all'):
        if mode == 'mnist':
            opts['digits'] = select_values('Digits', ['0','1','2'], interactive)
            t = choose('MNIST subset sizes', [('both','Both t=1 and t=2'),
                ('1','t=1 only'), ('2','t=2 only')], interactive=interactive)
            opts['ts'] = ['1','2'] if t == 'both' else [t]
        opts['rule'] = choose('MNIST candidate definition', [
            ('notebook-subsets','Recovered notebook computation (distinct marginal subsets)'),
            ('paper-multisets','Current manuscript definition (allows repeated marginals)')],
            interactive=interactive)
    if mode in ('sign','all'):
        if mode == 'sign':
            opts['classes'] = select_values('Gestures', ['A','C','D'], interactive)
        if interactive:
            while True:
                value = input('Minutes per sign optimization stage [60]: ').strip() or '60'
                if value.isdigit() and int(value) > 0:
                    opts['minutes'] = int(value); break
                print('Enter a positive whole number of minutes.')
        print('\nFour sign stages per class. At 60 minutes/stage, A/C/D have 12 hours of solver budgets plus preparation; stages may finish early.')
    return opts


def build_jobs(opts, run):
    jobs = []
    mode = opts['mode']
    def add(label, relative_script, args=(), solver=True):
        jobs.append(dict(label=label, script=str(PACKAGE/relative_script),
                         args=[str(a) for a in args], solver=solver))
    if mode == 'checks':
        add('01_scipy_smoke','code/smoke_test.py',['--json',run/'scipy_check.json'],False)
        add('02_gurobi_smoke','code/smoke_gurobi.py',['--json',run/'gurobi_check.json'])
        add('03_image_check','../Barycenter_Launchers/check_images.py',['--json',run/'image_check.json'],False)
        add('04_cached_table','scripts/run_sign.py',['cached-table','--output',run/'cached_table.json'],False)
    if mode in ('ellipse','all'):
        e = opts['ellipse']
        args = ['--mode', e if e in ('all','reference') else 'deterministic', '--output',run/'ellipses']
        if e != 'reference': args += ['--solver',opts['solver'],'--threads',THREADS]
        if e in ('t1','t2'): args += ['--t',e[-1]]
        add('ellipse_comparison','scripts/run_ellipses.py',args,
            e != 'reference' and opts['solver'] == 'gurobi')
    if mode in ('mnist','all'):
        for digit in opts['digits']:
            for t in opts['ts']:
                add('mnist_digit{}_t{}'.format(digit,t),'scripts/run_mnist.py',
                    ['--digit',digit,'--t',t,'--solver','gurobi','--candidate-rule',opts['rule'],'--output',run/'mnist'])
    if mode in ('sign','all'):
        seconds = opts['minutes']*60
        for c in opts['classes']:
            stages=[('t1_stage1','t1',None),('t1_stage2','t1','t1_stage1'),
                    ('hybrid_stage1','hybrid','t1_stage1'),('hybrid_stage2','hybrid','hybrid_stage1')]
            for stage, command, parent in stages:
                args=[command,'--class',c,'--time-limit',seconds,'--output-dir',run/(c+'_'+stage)]
                if parent:
                    args += ['--initial-support',run/(c+'_'+parent)/'S.npy',
                             '--initial-weights',run/(c+'_'+parent)/'z.npy']
                add(c+'_'+stage,'scripts/run_sign.py',args)
    if mode in ('classify','all'):
        for variant, stage in [('t1_7200','t1_stage2'),('hybrid3600','hybrid_stage1')]:
            args=['classify','--variant',variant,'--output-dir',run/('classification_'+variant)]
            if mode == 'all':
                for c in ['A','C','D']:
                    args += ['--model-dir',c+'='+str(run/(c+'_'+stage))]
            add('classification_'+variant,'scripts/run_sign.py',args,False)
    return jobs


def save_state(path, state):
    tmp=path.with_suffix('.tmp')
    tmp.write_text(json.dumps(state,indent=2)+'\n')
    tmp.replace(path)


def stop_child(child):
    if child is None or child.poll() is not None:
        return
    previous={s:signal.getsignal(s) for s in (signal.SIGINT,signal.SIGTERM,signal.SIGHUP)}
    for s in previous: signal.signal(s,signal.SIG_IGN)
    try:
        os.killpg(child.pid,signal.SIGINT)
        try:
            child.wait(timeout=5)
        except subprocess.TimeoutExpired:
            os.killpg(child.pid,signal.SIGTERM)
            try: child.wait(timeout=3)
            except subprocess.TimeoutExpired:
                os.killpg(child.pid,signal.SIGKILL); child.wait()
    except ProcessLookupError:
        child.wait()
    finally:
        for s,handler in previous.items(): signal.signal(s,handler)


def interrupt_queue(signum, frame):
    raise KeyboardInterrupt


def run_queue(opts, jobs, run):
    # One heavy queue at a time, even if several .command files are clicked.
    HERE.mkdir(exist_ok=True)
    lock=(HERE/'.run.lock').open('a+')
    try:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
    except BlockingIOError:
        print('Another Barycenter launcher is running. Use its Terminal window or stop it with Control-C.')
        return 2
    run.mkdir(parents=True,exist_ok=False)
    (run/'logs').mkdir()
    state=dict(status='running',started=datetime.now().isoformat(),python=str(PYTHON),
               package=str(PACKAGE),settings=opts,threads=THREADS,results=str(run),jobs=[])
    for job in jobs:
        state['jobs'].append(dict(job,status='pending'))
    save_state(run/'status.json',state)
    print('\nResults and logs: '+str(run))
    print('Jobs run one at a time. Keep this Terminal open. Control-C stops the queue.')
    print('Keep the laptop plugged in and the lid open. Idle sleep is prevented while the queue runs.\n')
    env=os.environ.copy()
    env.update(PYTHONUNBUFFERED='1',MPLCONFIGDIR=str(run/'.matplotlib'),
               XDG_CACHE_HOME=str(run/'.cache'),OMP_NUM_THREADS=str(THREADS),
               OPENBLAS_NUM_THREADS=str(THREADS),MKL_NUM_THREADS=str(THREADS),
               NUMEXPR_NUM_THREADS=str(THREADS))
    previous_signals={s:signal.getsignal(s) for s in (signal.SIGHUP,signal.SIGTERM)}
    for s in previous_signals: signal.signal(s,interrupt_queue)
    sleeper=None
    child=None
    current=None
    exitcode=0
    try:
        sleeper=subprocess.Popen(['/usr/bin/caffeinate','-i','-w',str(os.getpid())],
                                 stdout=subprocess.DEVNULL,stderr=subprocess.DEVNULL)
        for index,job in enumerate(jobs):
            current=state['jobs'][index]
            current.update(status='running',started=datetime.now().isoformat())
            save_state(run/'status.json',state)
            command=[str(PYTHON),'-u',str(HERE/'run_worker.py'),'--threads',str(THREADS)]
            if job['solver']: command+=['--solver']
            command += [job['script']]+job['args']
            current['command']=command
            print('\n[{}/{}] {}'.format(index+1,len(jobs),job['label']),flush=True)
            print(shlex.join(command),flush=True)
            logpath=run/'logs'/('{:02d}_{}.log'.format(index+1,job['label']))
            current['log']=str(logpath)
            save_state(run/'status.json',state)
            with logpath.open('w',buffering=1) as log:
                log.write(shlex.join(command)+'\n')
                child=subprocess.Popen(command,cwd=str(PACKAGE),env=env,
                    stdout=subprocess.PIPE,stderr=subprocess.STDOUT,
                    text=True,encoding='utf-8',errors='replace',bufsize=1,start_new_session=True)
                current['pid']=child.pid
                save_state(run/'status.json',state)
                for line in child.stdout:
                    print(line,end='',flush=True); log.write(line)
                code=child.wait()
            current.update(exit_code=code,finished=datetime.now().isoformat(),
                           status='completed' if code==0 else 'failed')
            save_state(run/'status.json',state)
            child=None
            if code:
                print('\nThis job failed. The queue has stopped. See '+str(logpath))
                state['status']='failed'; exitcode=code; break
        else:
            state['status']='completed'
            print('\nAll selected jobs completed. Results: '+str(run))
    except KeyboardInterrupt:
        print('\nStopping the current job; completed results are kept.',flush=True)
        stop_child(child)
        if current: current.update(status='interrupted',finished=datetime.now().isoformat())
        state['status']='interrupted'; exitcode=130
    except Exception as exc:
        stop_child(child)
        if current: current.update(status='failed',error=str(exc))
        state['status']='failed'; state['error']=str(exc); exitcode=1
        print('\nLauncher error: '+str(exc))
    finally:
        state['finished']=datetime.now().isoformat()
        save_state(run/'status.json',state)
        if sleeper is not None and sleeper.poll() is None:
            sleeper.terminate()
            sleeper.wait()
        fcntl.flock(lock,fcntl.LOCK_UN)
        lock.close()
        for s,handler in previous_signals.items(): signal.signal(s,handler)
    return exitcode


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode',choices=['menu','checks','ellipse','mnist','sign','classify','all'],default='menu')
    parser.add_argument('--defaults',action='store_true',help='Use each menu default; does not suppress execution.')
    parser.add_argument('--plan-only',action='store_true',help='Print job commands without running or writing anything.')
    args=parser.parse_args()
    if not PYTHON.is_file() or not (PACKAGE/'code/wb_bary_algorithms.py').is_file():
        parser.error('The configured Python or sibling closing_gap_reproduction package is missing.')
    print('\nWasserstein barycenter experiments — laptop launchers')
    print('Python: '+str(PYTHON)+'  |  MOSEK/Gurobi ellipse solvers  |  4 worker threads')
    mode=args.mode
    if mode=='menu':
        mode=choose('What would you like to run?',[
            ('checks','Check setup (small tests, under a minute)'),
            ('ellipse','Ellipses (Figure 2)'),('mnist','MNIST (Figure 3)'),
            ('sign','Sign-gesture training (Figures 4–6; many hours)'),
            ('classify','Fresh classification with archived barycenters (Table 1)'),
            ('all','All recovered experiments sequentially (can exceed a day)'),
            ('exit','Exit')],interactive=not args.defaults)
    if mode=='exit': return 0
    opts=settings(mode,not args.defaults)
    run=RUNS/(datetime.now().strftime('%Y%m%d_%H%M%S_%f')+'_'+mode)
    jobs=build_jobs(opts,run)
    if args.plan_only:
        print(json.dumps(dict(settings=opts,results=str(run),jobs=jobs),indent=2)); return 0
    return run_queue(opts,jobs,run)


if __name__=='__main__':
    try: sys.exit(main())
    except (KeyboardInterrupt,EOFError):
        print('\nExited before starting a queue.'); sys.exit(130)
