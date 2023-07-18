(ns openllm.components.model-selection.data
  (:require-macros [openllm.build :refer [slurp MODELS_DATA_JSON_PATH_MACRO]]))

(def models-data (-> "./src/generated/models-data.json"
                     (slurp ,) ;; see `openllm.build/slurp` to see how this magic works
                     (js/JSON.parse ,)
                     (js->clj ,)))

(def models
  {"chatglm" {:ids ["thudm/chatglm-6b"
                    "thudm/chatglm-6b-int8"
                    "thudm/chatglm-6b-int4"
                    "thudm/chatglm2-6b"
                    "thudm/chatglm2-6b-int4"]}
   "dolly-v2" {:ids ["databricks/dolly-v2-3b"
                     "databricks/dolly-v2-7b"
                     "databricks/dolly-v2-12b"]}
   "falcon" {:ids ["tiiuae/falcon-7b"
                   "tiiuae/falcon-40b"
                   "tiiuae/falcon-7b-instruct"
                   "tiiuae/falcon-40b-instruct"]}
   "flan-t5" {:ids ["google/flan-t5-small"
                    "google/flan-t5-base"
                    "google/flan-t5-large"
                    "google/flan-t5-xl"
                    "google/flan-t5-xxl"]}
   "gpt-neox" {:ids ["eleutherai/gpt-neox-20b"]}
   "mpt" {:ids ["mosaicml/mpt-7b"
                "mosaicml/mpt-7b-instruct"
                "mosaicml/mpt-7b-chat"
                "mosaicml/mpt-7b-storywriter"
                "mosaicml/mpt-30b"
                "mosaicml/mpt-30b-instruct"
                "mosaicml/mpt-30b-chat"]}
   "opt" {:ids ["facebook/opt-125m"
                "facebook/opt-350m"
                "facebook/opt-1.3b"
                "facebook/opt-2.7b"
                "facebook/opt-6.7b"
                "facebook/opt-66b"]}
   "stablelm" {:ids ["stabilityai/stablelm-tuned-alpha-3b"
                     "stabilityai/stablelm-tuned-alpha-7b"
                     "stabilityai/stablelm-base-alpha-3b"
                     "stabilityai/stablelm-base-alpha-7b"]}
   "starcoder" {:ids ["bigcode/starcoder"
                      "bigcode/starcoderbase"]}})
