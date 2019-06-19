## How to setup your docker in the NVIDIA DGX Box

* SSH into the machine and switch to the py-smtag directory
  ```bash
  ssh embo-dgx01
  cd /raid/lemberge/py-smtag
  ```

* Now we create a new image based on the Dockerfile, using [`docker build`](https://docs.docker.com/engine/reference/commandline/build/) we need to give it a name. In this example we are working from the master branch so we will call it `embo/py-smtag:master`.

  ```bash
  git checkout master
  docker build -t embo/py-smtag:master .
  ```

  If it worked you should see something like this at the end of the output:

  ```
  Successfully built 921a4fbc48d2
  Successfully tagged embo/py-smtag:master
  ```

  You could also list all your images by running [`docker images`](https://docs.docker.com/engine/reference/commandline/images/) and find it:

  ```bash
  docker images

  # REPOSITORY                  TAG                 IMAGE ID            CREATED             SIZE
  # embo/py-smtag               master              921a4fbc48d2        About an hour ago   9.78GB
  # ...
  ```

* Check in you `docker-compose.yml` file that the image for the _smtag_ service is properly configured. Since our new image is called `embo/py-smtag:master` your file should look like this:

  ```yml
  services:
    # ...
    smtag:
      image: embo/py-smtag:master
  ```

* Start up your multi-container Docker application by running [`docker-compose up`](https://docs.docker.com/compose/reference/up). We will add the `-d` flag to run it in detached mode. Running this will create a container for each service espcified in the `docker-compose.yml` file:

  ```bash
  docker-compose up -d
  # Starting py-smtag_neo4j_1 ... done
  # Starting py-smtag_smtag_1 ... done
  ```

  You could list all the container currently running using [`docker ps`](https://docs.docker.com/engine/reference/commandline/ps/). In my case I see this:

  ```bash
  docker ps
  # CONTAINER ID        IMAGE               COMMAND                  CREATED              STATUS              PORTS                     NAMES
  # f6ff5689709a        neo4j:3.1           "/sbin/tini -g -- /d…"   About a minute ago   Up About a minute   7473-7474/tcp, 7687/tcp   py-smtag_neo4j_1
  ```

  As you see, a container called `py-smtag_neo4j_1` corresponding to the _neo4j_ service is currently running.
  However there is no container for the _smtag_ service. The container was created when we run `docker-compose up`, but it inmediately stopped because it didn't have anything to do.


  This can be seen if we run use the `--all` flag to list also the docntainer that have stopped:

  ```
  docker ps --all

  # CONTAINER ID        IMAGE                              COMMAND                  CREATED             STATUS                     PORTS                     NAMES
  # e1a54f6415fa        embo/py-smtag:master               "/usr/local/bin/nvid…"   25 seconds ago      Exited (0) 5 seconds ago                             py-smtag_smtag_1
  # 0b00a95abc26        neo4j:3.1                          "/sbin/tini -g -- /d…"   25 seconds ago      Up 8 seconds               7473-7474/tcp, 7687/tcp   py-smtag_neo4j_1
  ```

* Let's use [`docker-compose run`](https://docs.docker.com/compose/reference/run/) to create a new container based on the _smtag_ service, and give it a simple dummy task: `ls -la`

  ```bash
  docker-compose run smtag ls -la

  # NVIDIA Release 19.05 (build 6411784)
  # PyTorch Version 1.1.0a0+828a6a3
  #
  # ...
  #
  # total 92
  # drwxrwxr-x 10 1001 1001 4096 Jun 19 09:38 .
  # drwxrwxrwx  1 root root 4096 Jun 19 09:06 ..
  # ...
  #
  ```
  Every time we run a command like this a new container is created, based on the specified image. When the command finishes the container stops, but it is not deleted. If we run the command again a new containers is created and stopped. For example if we run the same command 3 times and we list `--all` the containers:

  ```bash
  docker-compose run smtag ls -la
  docker-compose run smtag ls -la
  docker-compose run smtag ls -la

  docker ps --all
  # CONTAINER ID        IMAGE                              COMMAND                  CREATED             STATUS                         PORTS                     NAMES
  # a4c81571ad6c        embo/py-smtag:master               "/usr/local/bin/nvid…"   6 minutes ago       Exited (0) 6 minutes ago                                 py-smtag_smtag_run_6d5d5df48a0e
  # c79d6926d2bd        embo/py-smtag:master               "/usr/local/bin/nvid…"   6 minutes ago       Exited (0) 6 minutes ago                                 py-smtag_smtag_run_f071322c7bc8
  # 924026a4b968        embo/py-smtag:master               "/usr/local/bin/nvid…"   7 minutes ago       Exited (0) 6 minutes ago                                 py-smtag_smtag_run_d94a8f19ba72
  # d4a625d9b138        embo/py-smtag:master               "/usr/local/bin/nvid…"   7 minutes ago       Exited (0) 7 minutes ago                                 py-smtag_smtag_run_e7f450794d70
  # e1a54f6415fa        embo/py-smtag:master               "/usr/local/bin/nvid…"   About an hour ago   Exited (0) About an hour ago                             py-smtag_smtag_1
  # 0b00a95abc26        neo4j:3.1                          "/sbin/tini -g -- /d…"   About an hour ago   Up About an hour               7473-7474/tcp, 7687/tcp   py-smtag_neo4j_1
  ```


  As you can see, every time we execute `docker-compose run` we a new container gets created and kept. At this stage it is still possible to spin those containers again and recover any information from them, but this is mostly intended for debugging and should not be part of your normal workflow.

  This containers are typically rather slim, but they are obvisouly taking up some space (check `/var/lib/docker/`)

  Because of this, it is regarded as a good practice to inmediately delete all the containers that are supposed to be created to execute a short 1 time task. This could be done manually, using [`docker rm`](https://docs.docker.com/engine/reference/commandline/rm/) or by using the `--rm` flag when executing `docker-compose run`, like this:

  ```
  docker-compose run --rm smtag ls -la

  docker ps --all
  # CONTAINER ID        IMAGE                              COMMAND                  CREATED             STATUS                         PORTS                     NAMES
  # a4c81571ad6c        embo/py-smtag:master               "/usr/local/bin/nvid…"   34 minutes ago      Exited (0) 34 minutes ago                                py-smtag_smtag_run_6d5d5df48a0e
  # c79d6926d2bd        embo/py-smtag:master               "/usr/local/bin/nvid…"   34 minutes ago      Exited (0) 34 minutes ago                                py-smtag_smtag_run_f071322c7bc8
  # 924026a4b968        embo/py-smtag:master               "/usr/local/bin/nvid…"   34 minutes ago      Exited (0) 34 minutes ago                                py-smtag_smtag_run_d94a8f19ba72
  # d4a625d9b138        embo/py-smtag:master               "/usr/local/bin/nvid…"   34 minutes ago      Exited (0) 34 minutes ago                                py-smtag_smtag_run_e7f450794d70
  # e1a54f6415fa        embo/py-smtag:master               "/usr/local/bin/nvid…"   About an hour ago   Exited (0) About an hour ago                             py-smtag_smtag_1
  # 0b00a95abc26        neo4j:3.1                          "/sbin/tini -g -- /d…"   About an hour ago   Up About an hour               7473-7474/tcp, 7687/tcp   py-smtag_neo4j_1
  ```

  As you can see, the containers listed now are the same ones as before

* To completely shut down all running containers, and to removed all the stopped containers you can use [`docker-compose down`](https://docs.docker.com/compose/reference/down/)

  ```bash
  docker-compose down
  # Stopping py-smtag_neo4j_1 ... done
  # Removing py-smtag_smtag_run_6d5d5df48a0e ... done
  # Removing py-smtag_smtag_run_f071322c7bc8 ... done
  # Removing py-smtag_smtag_run_d94a8f19ba72 ... done
  # Removing py-smtag_smtag_run_e7f450794d70 ... done
  # Removing py-smtag_smtag_1                ... done
  # Removing py-smtag_neo4j_1                ... done
  # Removing network py-smtag_default
  ```

  If you try listing `--all` the containers again you should see an empty list:

  ```bash
  docker ps --all
  # CONTAINER ID     IMAGE     COMMAND     CREATED     STATUS     PORTS     NAMES
  ```
