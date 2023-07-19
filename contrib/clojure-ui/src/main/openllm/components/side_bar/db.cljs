(ns openllm.components.side-bar.db
  "The branch of the `app-db` that saves data related to the `side-bar` view. This
   mostly revolves around the model parameters.
   The path to this branch can be expressed as:
   *root -> components -> side-bar*"
   (:require [openllm.components.side-bar.model-selection.db :as model-selection-db]
             [cljs.spec.alpha :as s]))

(defn key-seq
  "Returns the key sequence to access the side-bar-db This is useful for
   `assoc-in` and `get-in`. The `more-keys` argument is optional and can be
   used to access a sub-key of the side-bar-db
   Returns the key sequence to access the side-bar-db"
  [& more-keys]
  (into [:components-db :side-bar-db] more-keys))

(def parameter-constraints
  "A map with parameter id's as keys and a vector of min and max values
   respectively as values."
  {::temperature [0.0 1.0]
   ::top_k [0 100]
   ::top_p [0.1 1.0]
   ::typical_p [0.1 1.0]
   ::epsilon_cutoff [0.0 1.0]
   ::eta_cutoff [0.0 1.0]
   ::diversity_penalty [0.0 5.0]
   ::repetition_penalty [0.0 5.0]
   ::encoder_repetition_penalty [0.0 5.0]
   ::length_penalty [0.0 5.0]
   ::num_beams [0 10]
   ::penalty_alpha [0.0 10.0]})

(defn get-validate-range-predicate
  "Returns a predicate that checks if the value is within the range of the
   parameter. The parameter is specified by the `type-predicate` argument.
   The predicate is a function that takes a value and returns true if the value
   is within the range of the parameter and false otherwise."
  [keyword type-predicate]
  (let [param (keyword parameter-constraints)]
    (s/and type-predicate
           #(<= (first param) % (second param)))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                Spec                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(s/def ::side-bar-open? boolean?)

(s/def ::temperature (get-validate-range-predicate ::temperature float?))
(s/def ::top_k (get-validate-range-predicate ::top_k int?))
(s/def ::top_p (get-validate-range-predicate ::top_p float?))
(s/def ::typical_p (get-validate-range-predicate ::typical_p float?))
(s/def ::epsilon_cutoff (get-validate-range-predicate ::epsilon_cutoff float?))
(s/def ::eta_cutoff (get-validate-range-predicate ::eta_cutoff float?))
(s/def ::diversity_penalty (get-validate-range-predicate ::diversity_penalty float?))
(s/def ::repetition_penalty (get-validate-range-predicate ::repetition_penalty float?))
(s/def ::encoder_repetition_penalty (get-validate-range-predicate ::encoder_repetition_penalty float?))
(s/def ::length_penalty (get-validate-range-predicate ::length_penalty float?))
(s/def ::num_beams (get-validate-range-predicate ::num_beams int?))
(s/def ::penalty_alpha (get-validate-range-predicate ::penalty_alpha float?))
(s/def ::max_new_tokens int?)
(s/def ::min_length int?)
(s/def ::min_new_tokens int?)
(s/def ::early_stopping boolean?)
(s/def ::max_time float?)
(s/def ::num_beam_groups int?)
(s/def ::use_cache boolean?)
(s/def ::model-config (s/keys :req [::temperature ::top_k
                                    ::top_p ::typical_p
                                    ::epsilon_cutoff ::eta_cutoff
                                    ::diversity_penalty ::repetition_penalty
                                    ::encoder_repetition_penalty ::length_penalty
                                    ::max_new_tokens ::min_length
                                    ::min_new_tokens ::early_stopping
                                    ::max_time ::num_beams
                                    ::num_beam_groups ::penalty_alpha
                                    ::use_cache]))

(s/def ::side-bar-db (s/keys :req-un [::side-bar-open?
                                      ::model-selection-db/model-selection-db
                                      ::model-config]))

(def initial-model-config
  "Very arbitrary. Should be fetched from metadata endpoint eventually." ;; TODO: fetch from metadata endpoint
  (array-map ::temperature 0.9
             ::top_k 50
             ::top_p 0.4
             ::typical_p 1.0
             ::epsilon_cutoff 0.0
             ::eta_cutoff 0.0
             ::diversity_penalty 0.0
             ::repetition_penalty 1.0
             ::encoder_repetition_penalty 1.0
             ::length_penalty 1.0
             ::max_new_tokens 2048
             ::min_length 0
             ::min_new_tokens 0
             ::early_stopping false
             ::max_time 0.0
             ::num_beams 1
             ::num_beam_groups 1
             ::penalty_alpha 0.0
             ::use_cache true))

(defn initial-db
  "Initial values for this branch of the app-db."
  []
  {:side-bar-open? true
   :model-selection-db (model-selection-db/initial-db)
   :model-config initial-model-config})


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;           Rich Comments            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(comment
  ;; check if initial-db is valid
  (s/valid? ::side-bar-db (initial-db)))
