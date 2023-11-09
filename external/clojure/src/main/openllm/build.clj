(ns openllm.build
  "This namespace contains build-time functions. These functions are run
   at compile time and are used to generate files that are later used by
   the application. For example, the `models-data.json` file is generated
   here. This file is later used to populate the Model-ID and Model-Type
   dropdowns with their respective entries.
   See `src/main/openllm/components/model_selection/data.cljs` for an example usage."
  (:require [clojure.java.shell :refer [sh]])
  (:refer-clojure :exclude [slurp]))

(def ^:private ^:const generic-io-error-msg-lol
  (str "It is strongly recommended to fix this error, otherwise the UI might not work as expected.\n"
       "Checks: Is openllm in your PATH?\n"
       "        Do you have sufficient priviledges?\n"
       "        Does the directory './src/generated/' exist?"))

(defn generate-models-json
  "Generates the `models-data.json` file from the `openllm models -o json`
   command. That file is later used to populate the Model-ID and Model-Type
   dropdowns with their respective entries.
   Runs before any compilation is done, as the stage is set to
   `:compile-prepare`.

   Returns the build-state as it was received."
  {:shadow.build/stage :compile-prepare}
  [build-state]
  (try
    (let [models-json (sh "openllm" "models" "-o" "json")]
      (spit "./src/generated/models-data.json" (:out models-json)))
    (catch Exception e
      (println "Failed to generate models-data.json file. Error: " e)
      (println generic-io-error-msg-lol)))
  build-state)


;; this macro is later used to read the generated `models-data.json` file
;; at compile time. See `src/main/openllm/components/model_selection/data.cljs`
;; for an example usage.
(defmacro slurp
  [file]
  (clojure.core/slurp file))
