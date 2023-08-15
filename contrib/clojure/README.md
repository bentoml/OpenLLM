# OpenLLM ClojureScript UI

## Prerequisites
* [Node.js](https://nodejs.org/en/) (v18.16.1)
* [npm](https://www.npmjs.com/) (v9.5.1)
* [Clojure](https://clojure.org/guides/getting_started) (v1.10.3)
* [OpenLLM](https://github.com/bentoml/OpenLLM) (latest)
Note: **The `openllm` executable must be in your `PATH`!** Alternatively, you could hack the `contrib\clojure-ui\src\main\openllm\build.clj` file to use the absolute path to the `openllm` executable. This should probably be possible via environment variables in the future.

## Recommended
* [Visual Studio Code](https://code.visualstudio.com/)
* [Calva](https://marketplace.visualstudio.com/items?itemName=betterthantomorrow.calva) (Plugin for VS Code)

# Developement Build
Open a terminal in the projects root and execute the following commands to start the development build:
```bash
cd dashboard/clojure
npm install
npm run dev
```
Or simply using hatch:
```bash
hatch run clojure-ui
```
You will now have a shadow-cljs instance running in the background, which will automatically compile the ClojureScript code and serve it to the browser. Since this is a `watch` build, the ClojureScript and Tailwind code will be recompiled and the browser will be refreshed whenever a file is changed.

## Connecting a REPL
The REPL is the most important tool for developing Clojure applications.

It is highly recommended to use *Visual Studio Code* + *Calva* for development of this project. Being able to evaluate code directly in the editor is a huge productivity boost and rich comments not only are an excellent way to document code, there are also a lot of useful ones in this codebase, i.e. for clearing persistent storage, dispatching various events and more.
I cannot stress the value of this enough, with Calva you can even evaluate a form directly into a comment (`Ctrl+Alt+C & C`).

That being said, if you prefer to use a different editor, the information on how to connect a REPL to the running shadow-cljs instance can be found in the [REPL section](https://shadow-cljs.github.io/docs/UsersGuide.html#_repl_2) of the shadow-cljs documentation. The nREPL port is printed to the console when starting the watch build and can also be found in the `.shadow-cljs/nrepl.port` file.

### VS Code + Calva:
To connect a REPL to the running shadow-cljs instance using *VS Code* + *Calva*, open the command palette (`Ctrl+Shift+P`) and search for "Calva: Connect to a running REPL server in the project" or use the shortcut `Ctrl+Shift+C & Ctrl+Shift+C`.

Then, select "shadow-cljs" from the dropdown menu. Another dropdown menu will appear, select ":app" from that menu. You should now be connected to the running shadow-cljs instance.

**Here is a list of the most important keyboard shortcuts for Calva:**
* `Ctrl+Shift+C & Enter` - Load/Evaluate current namespace & load it's dependencies
* `Ctrl+Enter` - Evaluate current form
* `Ctrl+Alt+C & C` - Evaluate current form into a comment

For more information on Calva, please consider reading the [Calva documentation](https://calva.io/finding-commands/).

## Source maps
Important to have a sane way to understand what is going on in the browser console.

Please refer to your browsers documentation on how to enable source maps. For Chrome, simply open the dev tools (`Ctrl+Shift+I`) and click the settings icon in the top right corner. Under "Sources" check the "Enable JavaScript source maps" checkbox, you should now correctly see the source code references in the browser console.

# Production Build
Run the following commands to build the production version of the dashboard:
```bash
cd contrib/clojure
npm install
npm run release
```
The compiled files will be located in the `contrib/clojure/public` directory. You can open the `index.html` file in your browser to view the dashboard.

TODO: Add a way to bundle the dashboard into a executable of some kind, which starts a webserver with the contents of th `contrib/clojure/public` directory.
