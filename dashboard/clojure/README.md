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
cd src/openllm_web_client
npm install
npm run dev
```

### VS Code + Calva:
```text
Ctrl+Shift+C & Ctrl+Shift+J
```
Then, select "deps.edn + shadow-cljs" from the dropdown menu. Another dropdown menu will appear, select ":cljs" from that menu. Finally, press the "Start a REPL and Connect (aka Jack-in)" button.
