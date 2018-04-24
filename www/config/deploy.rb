# config valid only for current version of Capistrano
lock "3.10.2"

set :application, "py-smtag"
set :repo_url, "git@github.com:source-data/py-smtag.git"

# Default branch is :master
# ask :branch, `git rev-parse --abbrev-ref HEAD`.chomp
# set :branch, 'master'
puts(fetch(:branch))

# Default deploy_to directory is /var/www/my_app_name
# set :deploy_to, "/var/www/my_app_name"

# Default value for :format is :airbrussh.
# set :format, :airbrussh

# You can configure the Airbrussh format using :format_options.
# These are the defaults.
# set :format_options, command_output: true, log_file: "log/capistrano.log", color: :auto, truncate: :auto

# Default value for :pty is false
# set :pty, true

# Default value for :linked_files is []
append :linked_files, ".env"

# Default value for linked_dirs is []
# append :linked_dirs, "node_modules"

# Default value for default_env is {}
# set :default_env, { path: "/opt/ruby/bin:$PATH" }

# Default value for local_user is ENV['USER']
# set :local_user, -> { `git config user.name`.chomp }

# Default value for keep_releases is 5
# set :keep_releases, 5

namespace :python do
  desc "Install requirements.txt"
  task :install_requirements do
    on roles [:app] do
      execute "source #{shared_path}/venv/bin/activate && pip install -r #{release_path}/www/requirements.txt"
    end
  end
end
after "deploy:updated", "python:install_requirements"
