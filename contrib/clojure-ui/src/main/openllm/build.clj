(ns openllm.build
  (:require [clojure.java.shell :refer [sh]])
  (:refer-clojure :exclude [slurp]))

(defn generate-models-json
  "Generates the `models-data.json` file from the `openllm models -o json`
   command. That file is later used to populate the Model-ID and Model-Type
   dropdowns with their respective entries.
   Runs before any compilation is done, as the stage is set to
   `:compile-prepare`.

   Returns the build-state as it was received."
  {:shadow.build/stage :compile-prepare}
  [build-state]
  (let [models-json (sh "openllm" "models" "-o" "json")]
    (spit "./src/generated/models-data.json"
          (:out models-json)))
  build-state)


;; this macro is later used to read the generated `models-data.json` file
;; at compile time. See `src/main/openllm/components/model_selection/data.cljs`
;; for an example usage.
(defmacro slurp
  [file]
  (clojure.core/slurp file))
