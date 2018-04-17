from flask import request
from smtag_api import app
from smtag_api import predictors

@app.route('/')
def hello_world():
    return 'Hello, World!'


# smtagapi_app = turbo.web.Application({
#     {"^/demo1$", turbo.web.StaticFileHandler, HOME.."/sdtag/index.html"},
#     {"^/demo2$", turbo.web.StaticFileHandler, HOME.."/www/demo2.html"},
#     {"^/demo3$", turbo.web.StaticFileHandler, HOME.."/www/demo3.html"},
#     {"^/smtag", SmtagHandler},
#     {"^/entity", EntityTaggerHandler},
#     {"^/role", RoleTaggerHandler},
#     {"^/tag", TaggerHandler},
#     {"^/panelize", PanelizerHandler},
#     {"^/shutdown$", ShutdownHandler},
#     {"^/hello", HelloWorld},
#     {"^/$", turbo.web.StaticFileHandler, HOME.."/index.html"},
#     {"^/.*/.*$", turbo.web.StaticFileHandler, HOME.."/sdtag/"}
# })


@app.route('/smtag', methods=['POST'])
def smtag():
    return run_predictor('smtag', request)

@app.route('/entity', methods=['POST'])
def entity():
    return run_predictor('entity', request)

@app.route('/role', methods=['POST'])
def role():
    return run_predictor('role', request)

@app.route('/tagger', methods=['POST'])
def tagger():
    return run_predictor('tagger', request)

@app.route('/panelize', methods=['POST'])
def panelize():
    return run_predictor('panelize', request)


def run_predictor(method, req):
    input_string = req.values.get("text", "")
    tag = req.values.get("tag", "sd-tag")
    format = req.values.get("format", "xml").lower()
    predictor = predictors.LuaCliPredictor(smtag_lua_cli_path=app.config['SMTAG_LUA_CLI_PATH'], torch_path=app.config['TORCH_PATH'])
    if method == "entity":
        return predictor.entity(input_string, format, tag)
    elif method == "smtag":
        return predictor.complete(input_string, format, tag)
    elif method == "role":
        return predictor.role(input_string, format, tag)
    elif method == "tagger":
        return predictor.tagger(input_string, format, tag)
    elif method == "panelize":
        return predictor.panelize(input_string, format, tag)
