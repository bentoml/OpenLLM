(ns openllm.build
  (:require [clojure.java.shell :refer [sh]]))

;;openllm models -o json
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
    (spit "www/generated/models-data.json"
          (:out models-json)))
  build-state)
