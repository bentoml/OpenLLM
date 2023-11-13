<div align="center">
    <h1 align="center">🧭 OpenLLM ClojureScript UI</h1>
    <p>Built using <a href="http://reagent-project.github.io/">reagent</a>, <a href="https://github.com/day8/re-frame">reframe</a>, <a href="https://tailwindcss.com/">Tailwind CSS</a>, and <a href="https://shadow-cljs.github.io/docs/UsersGuide.html">shadow-cljs</a></br></p>
    <i></i>
</div>

<img width="2880" alt="Screenshot 2023-08-16 at 03 34 34" src="https://github.com/bentoml/OpenLLM/assets/29749331/e0204483-7dc5-4694-be86-6ab554e7f992">

<br/>

## Installation

> [!NOTE]
> Currently, the LLMServer must be also running to use with the UI.
> Start any model server with `openllm start --cors`. For more information, see `openllm start -h`

> [!IMPORTANT]
> Make sure to have [Node.js](https://nodejs.org/en/) (v18.16.1)

```bash
hatch run ui:clojure
```

Access the UI at http://localhost:8420

### Connecting a REPL

The REPL is the most important tool for developing Clojure applications.

It is highly recommended to use _Visual Studio Code_ + _Calva_ for development of this project. Being able to evaluate code directly in the editor is a huge productivity boost and rich comments not only are an excellent way to document code, there are also a lot of useful ones in this codebase, i.e. for clearing persistent storage, dispatching various events and more.

That being said, if you prefer to use a different editor, the information on how to connect a REPL to the running shadow-cljs instance can be found in the [REPL section](https://shadow-cljs.github.io/docs/UsersGuide.html#_repl_2) of the shadow-cljs documentation. The nREPL port is printed to the console when starting the watch build and can also be found in the `.shadow-cljs/nrepl.port` file.

### VS Code + Calva:

To connect a REPL to the running shadow-cljs instance using _VS Code_ + _Calva_, open the command palette (`Ctrl+Shift+P`) and search for "Calva: Connect to a running REPL server in the project" or use the shortcut `Ctrl+Shift+C & Ctrl+Shift+C`.

Then, select "shadow-cljs" from the dropdown menu. Another dropdown menu will appear, select ":app" from that menu. You should now be connected to the running shadow-cljs instance.

> [!NOTE]
> Here is a list of keyboard shortcuts for Calva:
>
> - `Ctrl+Shift+C & Enter` - Load/Evaluate current namespace & load it's dependencies
> - `Ctrl+Enter` - Evaluate current form
> - `Ctrl+Alt+C & C` - Evaluate current form into a comment

For more information on Calva, please consider reading the [Calva documentation](https://calva.io/finding-commands/).

### Source maps

Important to have a sane way to understand what is going on in the browser console.

Please refer to your browsers documentation on how to enable source maps. For Chrome, simply open the dev tools (`Ctrl+Shift+I`) and click the settings icon in the top right corner. Under "Sources" check the "Enable JavaScript source maps" checkbox, you should now correctly see the source code references in the browser console.
