(ns openllm.components.model-selection.data
  (:require-macros [openllm.build :refer [slurp]])
  (:require [clojure.spec.alpha :as s]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                Spec                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(s/def ::vec-of-runtimes? (s/coll-of
                           (s/and string?
                                  #(or (= % "pt")
                                       (= % "flax")  ;; currently available runtimes
                                       (= % "tf")))
                           :kind vector?))

(s/def ::model_id (s/coll-of string? :kind vector?)) ;; model_id is a vector of all models for a given model_type
(s/def ::url string?)                                ;; url to the model's page
(s/def ::requires_gpu boolean?)                      ;; whether the model requires a GPU
(s/def ::runtime_impl ::vec-of-runtimes?)            ;; supported runtimes
(s/def ::installation string?)                       ;; installation instructions (pip command)

(s/def ::model-spec
  (s/keys :req-un [::model_id ::url ::requires_gpu   ;; the spec for a single model (aggregates all the above)
                   ::runtime_impl ::installation]))

(s/def ::models (s/map-of keyword? ::model-spec))    ;; map of all models


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;             Slurp Data             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(def models-data (-> "./src/generated/models-data.json"
                     (slurp ,) ;; see `openllm.build/slurp` to see how this magic works
                     (js/JSON.parse ,)
                     (js->clj , :keywordize-keys true)))

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
