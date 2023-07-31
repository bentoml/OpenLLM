(ns openllm.components.side-bar.model-selection.db
  "The branch of the `app-db` that saves data related to the model-selection view.
   This includes the current model selection, as well as the data for all available
   models.
   The path to this branch can be expressed as:
   *root -> components -> side-bar -> model-selection*"
  (:require [re-frame.core :as rf]
            [clojure.spec.alpha :as s]))

(defn key-seq
  "Returns the key sequence to access the model-selection-db This is useful for
   `assoc-in` and `get-in`. The `more-keys` argument is optional and can be
   used to access a sub-key of the model-selection-db
   Returns the key sequence to access the model-selection-db"
  [& more-keys]
  (into [:components-db :side-bar-db :model-selection-db] more-keys))

(def loading-text "Loading...")


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                Spec                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(s/def ::vec-of-runtimes? (s/coll-of
                           (s/and string?
                                  #(or (= % "pt")
                                       (= % "flax")  ;; all available runtimes
                                       (= % "tf")))
                           :kind vector?))

(s/def ::model_id (s/coll-of string? :kind vector?))                   ;; model_id is a vector of all models for a given model_type
(s/def ::url string?)                                                  ;; url to the model's page
(s/def ::requires_gpu boolean?)                                        ;; whether the model requires a gpu
(s/def ::runtime_impl ::vec-of-runtimes?)                              ;; supported runtimes
(s/def ::installation string?)                                         ;; installation instructions (pip command)
(s/def ::model-spec (s/keys :req-un [::model_id ::url ::requires_gpu   ;; the spec for a single model (aggregates all the above)
                                     ::runtime_impl ::installation]))
(s/def ::all-models #(or loading-text                                  ;; -- this is the case when the file with the model data has not been loaded yet by the ::set-model-data effect
                         (s/map-of keyword? ::model-spec)))            ;; map of all models

(s/def ::selected-model (s/keys :req-un
                                [::model-type #(or (keyword? %)        ;; currently selected model-id and model-type
                                                   (= % loading-text)) ;; -- same as above
                                 ::model-id string?]))

(s/def ::model-selection-db (s/keys :req-un [::all-models
                                             ::selected-model]))       ;; the spec of the model-selection-db

(defn initial-db
  "Initial values for this branch of the app-db.
   Triggers the loading of the model data by dispatching the `:slurp-model-data-json`
   event." 
  []
  (rf/dispatch [:slurp-model-data-json])
  (rf/dispatch [:fetch-metadata-endpoint])
  {:all-models loading-text ;; will be overwritten by the event dispatched above
   :selected-model {:model-type loading-text
                    :model-id loading-text}})


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;           Rich Comments            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(comment
  ;; check if initial-db is valid
  (s/valid? ::model-selection-db (initial-db)))
