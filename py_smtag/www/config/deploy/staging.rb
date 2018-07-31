role :app, %w{deploy@py-smtag-dev.embo.org}
set :deploy_to, "/var/www/py-smtag-dev.embo.org"
set :branch, proc { `git rev-parse --abbrev-ref HEAD`.chomp }
# set :default_env, { PATH: '/var/www/py-smtag-dev.embo.org/shared/venv/bin:$PATH' } # python venvs

# restart Passenger using `touch tmp/restart.txt`
set :passenger_restart_with_touch, true

server "py-smtag-dev.embo.org",
  user: "deploy",
  roles: %w{app},
  ssh_options: {
    user: "deploy", # overrides user setting above
    # keys: %w(~/.ssh/id_rsa), # doens't seem to work, you need to run `ssh-add ~/.ssh/id_rsa` on your shell instead
    forward_agent: true,
    auth_methods: %w(publickey)
    # password: "please use keys"
  }


# Passenger
# ============
# passenger once had only one way to restart: `touch tmp/restart.txt`
# Beginning with passenger v4.0.33, a new way was introduced: `passenger-config restart-app`
#
# The new way to restart was not initially practical for everyone,
# since for versions of passenger prior to v5.0.10,
# it required your deployment user to have sudo access for some server configurations.
#
# capistrano-passenger gives you the flexibility to choose your restart approach, or to rely on reasonable defaults.
#
# If you want to restart using `touch tmp/restart.txt`, add this to your config/deploy.rb:
#
#     set :passenger_restart_with_touch, true
#
# If you want to restart using `passenger-config restart-app`, add this to your config/deploy.rb:
#
#     set :passenger_restart_with_touch, false # Note that `nil` is NOT the same as `false` here
#
# If you don't set `:passenger_restart_with_touch`, capistrano-passenger will check what version of passenger you are running
# and use `passenger-config restart-app` if it is available in that version.
#
# If you are running passenger in standalone mode, it is possible for you to put passenger in your
# Gemfile and rely on capistrano-bundler to install it with the rest of your bundle.
# If you are installing passenger during your deployment AND you want to restart using `passenger-config restart-app`,
# you need to set `:passenger_in_gemfile` to `true` in your `config/deploy.rb`.
#

# server-based syntax
# ======================
# Defines a single server with a list of roles and multiple properties.
# You can define all roles on a single server, or split them:

# server "example.com", user: "deploy", roles: %w{app db web}, my_property: :my_value
# server "example.com", user: "deploy", roles: %w{app web}, other_property: :other_value
# server "db.example.com", user: "deploy", roles: %w{db}



# role-based syntax
# ==================

# Defines a role with one or multiple servers. The primary server in each
# group is considered to be the first unless any hosts have the primary
# property set. Specify the username and a domain or IP for the server.
# Don't use `:all`, it's a meta role.

# role :app, %w{deploy@example.com}, my_property: :my_value
# role :web, %w{user1@primary.com user2@additional.com}, other_property: :other_value
# role :db,  %w{deploy@example.com}



# Configuration
# =============
# You can set any configuration variable like in config/deploy.rb
# These variables are then only loaded and set in this stage.
# For available Capistrano configuration variables see the documentation page.
# http://capistranorb.com/documentation/getting-started/configuration/
# Feel free to add new variables to customise your setup.



# Custom SSH Options
# ==================
# You may pass any option but keep in mind that net/ssh understands a
# limited set of options, consult the Net::SSH documentation.
# http://net-ssh.github.io/net-ssh/classes/Net/SSH.html#method-c-start
#
# Global options
# --------------
#  set :ssh_options, {
#    keys: %w(/home/rlisowski/.ssh/id_rsa),
#    forward_agent: false,
#    auth_methods: %w(password)
#  }
#
# The server-based syntax can be used to override options:
# ------------------------------------
# server "example.com",
#   user: "user_name",
#   roles: %w{web app},
#   ssh_options: {
#     user: "user_name", # overrides user setting above
#     keys: %w(/home/user_name/.ssh/id_rsa),
#     forward_agent: false,
#     auth_methods: %w(publickey password)
#     # password: "please use keys"
#   }
