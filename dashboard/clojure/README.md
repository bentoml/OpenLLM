## Prerequisites
### Required
* [Node.js](https://nodejs.org/en/) (v18.16.1)
* [npm](https://www.npmjs.com/) (v9.5.1)
* [Clojure](https://clojure.org/guides/getting_started) (v1.10.3)

### Optional
* [VS Code](https://code.visualstudio.com/)
* [Calva](https://marketplace.visualstudio.com/items?itemName=betterthantomorrow.calva)

## Developement Build
### Raw terminal:
```bash
cd dashboard/clojure
npm install
npm run dev
```
Or simply using hatch:
```bash
hatch run clojure-ui
```
Please refer to external resources on how to connect a REPL to the running shadow-cljs instance. A good resource to start out is the [REPL section](https://shadow-cljs.github.io/docs/UsersGuide.html#_repl_2) in the shadow-cljs manual.

### VS Code + Calva:
```text
Ctrl+Shift+C & Ctrl+Shift+J
```
Then, select "deps.edn + shadow-cljs" from the dropdown menu. Another dropdown menu will appear, select ":cljs" from that menu. Finally, press the "Start a REPL and Connect (aka Jack-in)" button.
