# Snellius

## Basic info

- The Dutch National supercomputer hosted at SURF.
- A cluster of heterogeneous nodes built by Lenovo, containing predominantly AMD technology.
- Peak performance of 14 petaflop/s.
- To login via the terminal from a white-listed IP: `ssh user@snellius.surf.nl`
- From abroad or not white-listed-IP, there is a separate login server: doornode.surfsara.nl (thus using `ssh user@doornode.surfsara.nl`). Use your usual login and password, and select 'Snellius'. Please note that you cannot copy files or use X11 when using the door node.
- You can find the blocked ip-address when logging in to Snellius using `ssh -v [login]@snellius.surf.nl`
- To go to shared 3dVac folder: `cd /projects/0/einf2380/`
- The interactive nodes only accept (GSI-)ssh connections from known, white-listed IP, ranges.
- type `accinfo` on terminal to get general infos.

### How to connect to the Snellius system from abroad: see [here](https://servicedesk.surfsara.nl/wiki/pages/viewpage.action?pageId=30660265)

## Creating and running jobs

### The batch system

- It's a tool to distribute computational tasks over the available nodes.
- You prepare a set of commands (a 'job') and store them in the job script file, that the computer will execute later.
- When and where a job is executed is determined by the job scheduler.

### The job scheduler

- When you submit a job, it enters a job queue. A scheduler reads the job requirements from all submitted job scripts and determines when, and on which nodes these jobs should run.

### Writing a job script

- A job script generally consists of the following parts.
  1. Specification of the job requirements for the batch system (number of nodes, expected runtime, etc).
  2. Loading of modules needed to run your application. *not always required*
  3. Preparing input data (e.g. copying input data to from your home to scratch, preprocessing). *not always required*
  4. Running your application.
  5. Aggregating output data (e.g. post-processing, copying data from scratch to your home). *not always required*

### Script example

Common options that can be passed to the SLURM batch system, either on the `sbatch` command line, or as an `#SBATCH` attribute:

- -N [value]
- -n [value]
- -c [value]
- --tasks-per-node=[value]
- --cpus-per-task=[value]
- --mem=[value]
- --constraint=[value
- -t [HH:MM:SS] or [MM:SS] or [minutes]
- -p gpuGPU
- --requeue

All available SBATCH flags can be seen typing `sbatch --help` on the terminal. 

#### Single node, serial program example

```
#!/bin/bash
#Set job requirements
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --partition=thin
#SBATCH --time=01:00:00
 
#Execute program located in $HOME
$HOME/my_serial_program
```

In particular ... 

`#!/bin/bash` defines the interpreter for the job script with a shebang. Here we use Bash.

`#SBATCH --nodes=1` defines the number of nodes you need.

`#SBATCH --ntasks=1` sets the number of tasks per node. For example, MPI programs will use this number to determine how many processes to run per node.

`#SBATCH --cpus-per-task=32` the default will allocate 1 cpu per task. If you would like to advise the Slurm controller that job steps will require a different number of processors per task, use this option.

`#SBATCH --partition=thin` if you don't specify a partition, your job will be scheduled on the normal partition by default. The available compute partitions that batch jobs can be submitted to can be listed by issuing `sinfo` command. To get a summary of the available partitions, use `sinfo -s`. The available partitions of Snellius are described [here](https://servicedesk.surfsara.nl/wiki/display/WIKI/Snellius+usage+and+accounting).

`#SBATCH --time=01:00:00` sets the duration for which the nodes remain allocated to you (known as the wall clock time).

`$HOME/my_serial_program` runs a single instance of a serial (i.e. non-parallel) program, taking a single input file as an argument. 

### Loading modules

- `module avail` to see which modules are available.
- `module spider` to find all possible modules and extensions. 
- `module load [module_name]` to load a module. 
- Default modules are disabled on Snellius and Lisa. You need to specify the full version of the module you want to load (e.g: Python/3.9.5-GCCcore-10.3.0).

### Final job script

```
#!/bin/bash
#Set job requirements
#SBATCH -n 16
#SBATCH -t 5:00
 
#Loading modules
module load 2021
module load Python/3.9.5-GCCcore-10.3.0
 
#Copy input file to scratch
cp $HOME/big_input_file "$TMPDIR"
 
#Create output directory on scratch
mkdir "$TMPDIR"/output_dir
 
#Execute a Python program located in $HOME, that takes an input file and output directory as arguments.
python $HOME/my_program.py "$TMPDIR"/big_input_file "$TMPDIR"/output_dir
 
#Copy output directory from scratch to home
cp -r "$TMPDIR"/output_dir $HOME
```

Note that The **scratch disk** provides temporary storage that is much faster than the home file system. This is particularly important if, for example, you launch 16 processes on a single node, that each need to read in files (or worse, you launch an MPI program over 10 nodes, with 16 processes per node). Your input files will be read 16 (or, in the MPI case: 160) times. In case you have 1 or 2 tiny input files, it may be acceptable to read these directly from the home file system. If you have many files and/or very large files, however, this will put too high a load on the home file system - and slow down your job considerably.

The solution is to copy your input data to the local scratch disk of each node before starting your application. Then, each of the 16 processes on that node can read the input data from the local scratch disk. For the single-node example, you reduce the number of file reads from the home file system from 16 to 1 (i.e. only the copy operation).
