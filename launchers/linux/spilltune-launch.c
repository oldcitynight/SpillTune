/* SPDX-License-Identifier: GPL-3.0-or-later
 * Own Linux launcher source. No Python interpreter or dependency is embedded.
 * See COPYING and LICENSE.md for the project license scope.
 */
#define _POSIX_C_SOURCE 200809L
#include <errno.h>
#include <limits.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <sys/stat.h>
#include <unistd.h>

static const char bootstrap[] =
"import importlib,json,os,pathlib,runpy,struct,sys,sysconfig\n"
"app=pathlib.Path(sys.argv.pop(1)).resolve()\n"
"check=sys.argv.pop(1)=='check'\n"
"sys.path.insert(0,str(app))\n"
"sys.argv[0]='spilltune'\n"
"try:\n"
" if sys.version_info<(3,11): raise RuntimeError('Python 3.11+ required')\n"
" import ram_vram_finetune as own\n"
" if not pathlib.Path(own.__file__).resolve().is_relative_to(app): raise RuntimeError('own tool payload mismatch')\n"
" if check:\n"
"  deps={}\n"
"  for name in ('torch','safetensors','transformers'):\n"
"   module=importlib.import_module(name);deps[name]={'version':module.__version__,'file':str(pathlib.Path(module.__file__).resolve())}\n"
"  import torch\n"
"  print(json.dumps({'status':'EXTERNAL_ENVIRONMENT_IMPORTS_OK','selection':os.environ.get('SPILLTUNE_LAUNCHER_SELECTION'),'python':sys.executable,'prefix':sys.prefix,'python_version':list(sys.version_info[:3]),'soabi':sysconfig.get_config_var('SOABI'),'bits':8*struct.calcsize('P'),'tool_version':own.__version__,'tool_file':own.__file__,'dependencies':deps,'cuda_build':torch.version.cuda,'cuda_initialized':torch.cuda.is_initialized(),'installation_performed':False},ensure_ascii=False))\n"
" else: runpy.run_module('ram_vram_finetune.cli',run_name='__main__')\n"
"except (ImportError,OSError,RuntimeError) as error:\n"
" print('spilltune launcher: '+type(error).__name__+': '+str(error)+'; selected Python='+sys.executable+'; prepare compatible dependencies in this exact environment. Nothing was installed.',file=sys.stderr)\n"
" sys.exit(2)\n";

static int regular(const char *path) {
    struct stat st;
    return stat(path, &st) == 0 && S_ISREG(st.st_mode);
}

int main(int argc, char **argv) {
    const char *python = NULL, *app = NULL;
    char executable[PATH_MAX], app_buffer[PATH_MAX], check_file[PATH_MAX], python_buffer[PATH_MAX];
    const char *selection = "argument";
    int check = 0, first = argc;
    for (int i = 1; i < argc; ++i) {
        if (strcmp(argv[i], "--") == 0) { first = i + 1; break; }
        if (strcmp(argv[i], "--python") == 0 && i + 1 < argc) { python = argv[++i]; continue; }
        if (strcmp(argv[i], "--app") == 0 && i + 1 < argc) { app = argv[++i]; continue; }
        if (strcmp(argv[i], "--check") == 0) { check = 1; continue; }
        first = i; break; /* Keep the tool CLI unchanged; '--' is optional. */
    }
    if (python == NULL) {
        python = getenv("SPILLTUNE_PYTHON"); selection = "SPILLTUNE_PYTHON";
        if (python == NULL || python[0] == '\0') {
            const char *prefix = getenv("VIRTUAL_ENV"); selection = "VIRTUAL_ENV";
            if (prefix == NULL || prefix[0] == '\0') { prefix = getenv("CONDA_PREFIX"); selection = "CONDA_PREFIX"; }
            if (prefix != NULL && prefix[0] != '\0') {
                if (snprintf(python_buffer, sizeof(python_buffer), "%s/bin/python", prefix) >= (int)sizeof(python_buffer)) return 2;
                python = python_buffer;
            } else { python = "/usr/bin/python3"; selection = "system Python"; }
        }
    }
    if (python == NULL || python[0] != '/' || !regular(python) || access(python, X_OK) != 0) {
        fprintf(stderr, "spilltune launcher: --python must name an existing absolute executable; no environment was selected or installed automatically.\n");
        return 2;
    }
    if (app == NULL) {
        ssize_t length = readlink("/proc/self/exe", executable, sizeof(executable) - 1);
        if (length <= 0 || length >= (ssize_t)sizeof(executable) - 1) {
            fprintf(stderr, "spilltune launcher: cannot locate own executable; use --app with an absolute own payload directory.\n"); return 2;
        }
        executable[length] = '\0'; char *slash = strrchr(executable, '/');
        if (slash == NULL) return 2;
        *slash = '\0';
        if (snprintf(app_buffer, sizeof(app_buffer), "%s/app", executable) >= (int)sizeof(app_buffer)) return 2;
        if (snprintf(check_file, sizeof(check_file), "%s/ram_vram_finetune/__init__.py", app_buffer) >= (int)sizeof(check_file)) return 2;
        if (!regular(check_file) && snprintf(app_buffer, sizeof(app_buffer), "%s/../share/spilltune/app", executable) >= (int)sizeof(app_buffer)) return 2;
        app = app_buffer;
    }
    if (app[0] != '/' || snprintf(check_file, sizeof(check_file), "%s/ram_vram_finetune/__init__.py", app) >= (int)sizeof(check_file) || !regular(check_file)) {
        fprintf(stderr, "spilltune launcher: own tool payload missing; reinstall the own tool or select --app explicitly.\n"); return 2;
    }
    if (setenv("SPILLTUNE_LAUNCHER_SELECTION", selection, 1) != 0) return 2;
    int extra = argc - first;
    char **command = calloc((size_t)extra + 10, sizeof(char *));
    if (command == NULL) return 2;
    int n = 0;
    command[n++] = (char *)python;
    command[n++] = "-I"; command[n++] = "-B";
    command[n++] = "-X"; command[n++] = "utf8";
    command[n++] = "-c"; command[n++] = (char *)bootstrap;
    command[n++] = (char *)app; command[n++] = check ? "check" : "run";
    for (int i = first; i < argc; ++i) command[n++] = argv[i];
    command[n] = NULL;
    execv(python, command);
    fprintf(stderr, "spilltune launcher: selected Python failed to execute: %s\n", strerror(errno));
    free(command); return 127;
}
