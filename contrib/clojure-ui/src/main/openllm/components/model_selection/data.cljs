(ns openllm.components.model-selection.data
  (:require-macros [openllm.build :refer [slurp]])
  (:require [clojure.spec.alpha :as s]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                Spec                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(s/def ::vec-of-runtimes? (s/coll-of
                           (s/and string?
                                  #(or (= % "pt")
                                       (= % "flax")  ;; all available runtimes
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
(def ^:const models-data (-> "./src/generated/models-data.json"
                             (slurp ,) ;; see `openllm.build/slurp` to see how this sorcery works
                             (js/JSON.parse ,)
                             (js->clj , :keywordize-keys true)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;            Convenience             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn model-types []
  "Returns a list of all `model-types`."
  (keys models-data))

(defn model-ids [model-type]
  "Returns a list of all `model-ids` for a given `model-type`."
  (get-in models-data [model-type :model_id]))
