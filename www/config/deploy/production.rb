role :app, %w{deploy@sdtag.com}
set :deploy_to, "/var/www/sdtag.embo.org"
set :branch, :master

# restart Passenger using `touch tmp/restart.txt`
set :passenger_restart_with_touch, true

server "py-smtag-dev.embo.org",
  user: "deploy",
  roles: %w{app},
  ssh_options: {
    user: "deploy",
    forward_agent: true,
    auth_methods: %w(publickey)
  }
