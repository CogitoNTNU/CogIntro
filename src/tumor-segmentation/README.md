# High Performance Computing SLURM Course

High Performance Computing (HPC) is an incredibly powerful tool for developers—especially for AI engineers training large models. You can achieve strong AI models through three key ingredients:

1. **Lots of data**
1. **Powerful compute resources**
1. **Effective learning algorithms**

While HPC is extremely useful, many users find that you can run into issues you wouldn’t see on a local machine, and the errors can be harder to debug.

______________________________________________________________________

## Connectivity

### SSH to IDUN (outside NTNU network)

```bash
ssh -J <username>@login.stud.ntnu.no <username>@idun-login1.hpc.ntnu.no
```

### SSH to IDUN (inside NTNU network)

```bash
ssh <username>@idun-login1.hpc.ntnu.no
```

## Checking your Quota

Check all groups’ quotas:

```bash
idun-slurm-quota
```

Check your own group associations:

```bash
sacctmgr show assoc where user=<username> format=account --parsable
```

## Key SLURM Commands

Submit a job:

```bash
sbatch <path-of-job.slurm>
```

Cancel a job:

```bash
scancel <job-id>
```

Get all your pending or running jobs

```bash
squeue --me
```

List running/pending jobs for user:

```bash
squeue -u <username>
```

Watch running job execution:

```bash
watch tail -20 <path-to-your-job-output-file>
```

## Quality of Life Commands by Martin W. Holtmon

```bash
alias get_quota='lfs quota -u $(whoami) /cluster'
alias getacc='sacctmgr show assoc where user=$(whoami) format=account --parsable2 --noheader'
alias getdefaultacc='sacctmgr show user $(whoami) format=defaultaccount --parsable2 --noheader'
alias gowork='cd /cluster/work/$(whoami)/'
alias past_jobs='sacct -X --format=JobID,Jobname%30,state,time,elapsed,nnodes,ncpus,nodelist,AllocTRES%50'
alias queue='watch -n 5 "squeue --me --format=\"%.18i %.9P %.30j %.8u %.8T %.10M %.9l %.6D %.19S %R\""'
```

## Additional Resources

- Most user-friendly tutorial: [IDUN Tutorial from the Perspective of a Master’s Student (2024 edition)](https://www.hpc.ntnu.no/idun/documentation/idun-tutorial-from-the-perspective-of-a-masters-student-2024-edition/)
- Complete SLURM Docs: https://slurm.schedmd.com
- IDUN & Hardware Specs: https://www.hpc.ntnu.no/idun/
- IDUN Status Page (NTNU intranet only): http://idun.hpc.ntnu.no/
- Connect VS Code to IDUN: https://www.hpc.ntnu.no/idun/documentation/visual-studio-code-connect-to-idun-hpc-cluster/
- General Documentation: https://www.hpc.ntnu.no/idun/documentation/
