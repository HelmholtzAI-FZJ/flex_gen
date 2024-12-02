# sc_venv_template_hydra
## Setup Virtual Environmnet
This is adapted from sc_venv_template https://gitlab.jsc.fz-juelich.de/kesselheim1/sc_venv_template

This hydra version means to better manage the venv across the machine, if you need run same project in different machine.

As introduced in [sc_venv_template](https://gitlab.jsc.fz-juelich.de/kesselheim1/sc_venv_template), you need to 

- Edit `config.sh` to change name an location of the venv if required.
- Edit `modules.sh` to change the modules loaded prior to the creation of the venv.
- Edit `install_requirements.sh to change the packages to be installed during setup.


And then create the environment with `setup.sh`, but do it on each machine
- On machine A, `sh ./sc_venv_template/setup.sh`
- On machine B, `sh ./sc_venv_template/setup.sh`
- And so on


After all, there will several venv created in `sc_venv_template` which will look like:
```bash
- sc_venv_template
    - venv_jurecadc
    - venv_juwelsbooster
    - venv_juwel
    - ...
```

## Use Virtual Environmnet
Same as in original templete, you just need to use `activate.sh` to activate you virtual enviroment, does not matter about machine: `source ./sc_venv_template/activate.sh`

Put it on sbatch scripts, it will automatically load the correct venv.

# Warning!!
This scripts can only make sure you have exactly same enviroment at setup.
If you install new packages in venv_MACHINE_A, you need to manually do it for venv_MACHINE_B, and others.
