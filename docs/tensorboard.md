# On your local machine
Install Tensorboard by installing Tensorflow

    pip install --upgrade tensorflow

Run `meta.py` once to have some data

    python meta.py

Run tensorboard

    tensorboard --logdir runs


# Setting up Tensorboard on a Linux server

Dependencies:

* nginx
* python 3.6

Assumptions that may differ on your system:

* We use a local user called _ubuntu_ who has its home in `/home/ubuntu`
* The tensorboard website will be accessed via `http://tensorboard.sdtag.net`

## Setup the project and run tensorboard once
As a first step and to make sure we have all the moving pieces working together we are goint to install py-smtag, run it once to generate some data and then run a on-demand tensorboard server to display the results.

Go to the home of your user and clone the github project

    cd /home/ubuntu
    git clone git@github.com:source-data/py-smtag.git
    cd py-smtag
    python3 -m venv .venv
    source .venv/bin/activate
    pip install -r requirements.txt
    pip install --upgrade tensorflow

We will manually run tensorboard once to make sure everything works fine.

First, let's generate some dummy data that tensorboard can display:

    # unzip the demo data
    unzip demo_data
    mkdir data
    mv test_* data

    # run meta
    python meta.py

Then you can manually launch your tensorboard:

    tensorboard --logdir runs

Now you can open your browser and point it to http://tensorboard.sdtag.net:6006

You should be able to see the Tensorboard interface.

The problem of this workflow is that when you close your console you also close tensorboard webserver. You could use tmux or run the process in the background, but what happens if the tensorboard process dies? or tmux? What if we want to run on a standard web port instead of 6006? Or what if we want to add SSL? If we are going to do it, let's do it right! :D

## Running tensorboard as a proper website
Now we will see how to create service that automatically launches tensorboard when the machine starts up, and will also monitor the process and relaunch in case it dies.

Next we will use NGINX as a web server to proxy to the tensorboard service. This will allow things (not covered here) such as:

* running on standard ports
* handling different CNAMEs
* adding SSL and running only over `https`


### Setup the hosting directory
Following standard web practices we will create a new directory under `/var/www` to host the tensorboard website. For normal websites this step feels more natural, here we are just looking for a place to create our python virtual environment and install tensorboard.

Note: our AMI instance didn't have a *www* user, so I created one: `sudo adduser www`

```sh
sudo su
mkdir -p /var/www/tensorboard.sdtag.net
chown www:www /var/www/tensorboard.sdtag.net
su www
cd /var/www/tensorboard.sdtag.net
python3 -m venv venv
source venv/bin/activate
pip3 install --upgrade tensorflow
```

If we peek into the virtual environment that we created we will see that by installing tensorflow we also installed the binary for tensorboard:

```sh
ls -la /var/www/tensorboard.sdtag.net/venv/bin | grep tensorboard
```

### Adding a systemd service

Systemd is an init system and system manager that has become the standar in Linux machines. We will use it to treat the tensorboard backend as a service in our machine, ensuring it will always be up and running even after every reboot, or in case the process dies. As reference I recommend reading [this](https://www.digitalocean.com/community/tutorials/systemd-essentials-working-with-services-units-and-the-journal) and [this](https://www.digitalocean.com/community/tutorials/how-to-use-systemctl-to-manage-systemd-services-and-units).

The next steps need to be executed as root unless otherwise specified.

Create a new systemd service

    cd /etc/systemd/system
    touch tensorboard.sdtag.net.service
    systemctl edit --full tensorboard.sdtag.net.service

And paste the following

    [Unit]
    Description=Tensorboard backend for http://tensorboard.sdtag.net
    After=syslog.target network.target nginx.service

    [Service]
    ExecStart=/home/ubuntu/py-smtag/.venv/bin/tensorboard --logdir /home/ubuntu/py-smtag/runs
    Restart=always
    RestartSec=10                       # Restart service after 10 seconds if node service crashes
    StandardOutput=syslog               # Output to syslog
    StandardError=syslog                # Output to syslog
    SyslogIdentifier=tensorboard

    [Install]
    WantedBy=multi-user.target

Then let's check the status, make sure it auto-launches after every reboot, and manually launch it for the first time.

    # Let's verify that the service is recognized and disabled
    systemctl is-enabled tensorboard.sdtag.net.service
    # It should also be inactive
    systemctl is-active tensorboard.sdtag.net.service
    # Let's enable, ie, ensure it launches automatically after rebooting the machine
    systemctl enable tensorboard.sdtag.net.service
    # And finally, let's manually start it
    systemctl start tensorboard.sdtag.net.service
    # You can stop it by running
    systemctl stop tensorboard.sdtag.net.service


### Configure NGINX

NGINX is an open source high performance web server that has become a standard on web development in the last decade. We will be using it as a reverse proxy to the Tensorboard backend. It can allow us to implement things like SSL (not covered here), IP whitelisting or Basic Authentication.

The next steps need to be executed as root unless otherwise specified.

Create the configuration file for the site

    # create the config file
    touch /etc/nginx/sites-available/tensorboard.sdtag.net
    # enable it
    ln -s /etc/nginx/sites-available/tensorboard.sdtag.net /etc/nginx/sites-enabled/
    # edit it
    vim /etc/nginx/sites-available/tensorboard.sdtag.net

And paste the following in it:

    server {
      listen 80;
      listen [::]:80;
      server_name tensorboard.sdtag.net;

      auth_basic "Restricted Content";
      auth_basic_user_file /etc/nginx/conf.d/htpasswd;


      # allow 172.21.0.0/16; # only allow connections from a specific ip or ip range
      # deny all; # deny the rest of connections

      location / {
        proxy_pass http://127.0.0.1:6006;
      }
    }

Notice it is password protected using HTTP Basic Auth. The user/password file is stored under `/etc/nginx/conf.d/htpasswd`. Let's create this file

    # make sure you have apache utils available
    apt install apache2-utils
    # generate a password for the user 'tensorboard'
    touch /etc/nginx/conf.d/htpasswd
    htpasswd /etc/nginx/conf.d/htpasswd tensorboard

And finally we need to restart nginx for it to pick the changes

    sudo service nginx restart

Now you should be able to open your browser and visit [http://tensorboard.sdtag.net](http://tensorboard.sdtag.net). You will be asked for your username and password we just created.











