# Run PyCUTEST and CUTEst in container

## 1. Install docker & pipenv 

## 2. Lock pipenv files to pre-install dependencies
```
cd docker
pipenv lock
```
## 3. Run docker-shell script to enter docker container 
```
sh docker-shell.sh

```
## 4. Run Example test file
```
python play.py
```