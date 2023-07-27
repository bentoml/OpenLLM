(ns openllm.components.side-bar.model-params.db
  "The branch of the `app-db` that saves data related to the model-params-db view.
   This includes all the configuration parameters for the models.
   The path to this branch can be expressed as:
   *root -> components -> side-bar -> model-params*"
  (:require [clojure.spec.alpha :as s]))

(defn key-seq
  "Returns the key sequence to access the model-params-db This is useful for
   `assoc-in` and `get-in`. The `more-keys` argument is optional and can be
   used to access a sub-key of the model-params-db
   Returns the key sequence to access the model-params-db"
  [& more-keys]
  (into [:components-db :side-bar-db :model-params-db] more-keys))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;                Spec                ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(def parameter-meta-data
  "A map with parameter id's as keys and some metadata for easier rendering." 
   {:temperature                  {:display-type :slider   :type-pred float?   :advanced-opt false :val-constraint [0.0 1.0]}
    :top_k                        {:display-type :slider   :type-pred int?     :advanced-opt false :val-constraint [0 100]}
    :top_p                        {:display-type :slider   :type-pred float?   :advanced-opt false :val-constraint [0.1 1.0]}
    :typical_p                    {:display-type :slider   :type-pred float?   :advanced-opt false :val-constraint [0.1 1.0]}
    :epsilon_cutoff               {:display-type :slider   :type-pred float?   :advanced-opt true  :val-constraint [0.0 1.0]}
    :eta_cutoff                   {:display-type :slider   :type-pred float?   :advanced-opt true  :val-constraint [0.0 1.0]}
    :diversity_penalty            {:display-type :slider   :type-pred float?   :advanced-opt true  :val-constraint [0.0 5.0]}
    :repetition_penalty           {:display-type :slider   :type-pred float?   :advanced-opt true  :val-constraint [0.0 5.0]}
    :encoder_repetition_penalty   {:display-type :slider   :type-pred float?   :advanced-opt true  :val-constraint [0.0 5.0]}
    :length_penalty               {:display-type :slider   :type-pred float?   :advanced-opt true  :val-constraint [0.0 5.0]}
    :num_beams                    {:display-type :field    :type-pred int?     :advanced-opt true  :val-constraint [0 10]}
    :penalty_alpha                {:display-type :slider   :type-pred float?   :advanced-opt true  :val-constraint [0.0 10.0]}
    :max_new_tokens               {:display-type :field    :type-pred int?     :advanced-opt true  :val-constraint [0 ##Inf]}
    :min_length                   {:display-type :field    :type-pred int?     :advanced-opt true  :val-constraint [0 ##Inf]}
    :min_new_tokens               {:display-type :field    :type-pred int?     :advanced-opt true  :val-constraint [0 ##Inf]}
    :early_stopping               {:display-type :binary   :type-pred boolean? :advanced-opt true  :val-constraint [true false]}
    :max_time                     {:display-type :field    :type-pred float?   :advanced-opt true  :val-constraint [0.0 ##Inf]}
    :num_beam_groups              {:display-type :field    :type-pred int?     :advanced-opt true  :val-constraint [0 ##Inf]}
    :use_cache                    {:display-type :binary   :type-pred boolean? :advanced-opt true  :val-constraint [true false]}})

(defn get-validate-range-predicate
  "Returns a predicate that checks if the value is within the range of the
   parameter. The parameter is specified by the `type-predicate` argument.
   The predicate is a function that takes a value and returns true if the value
   is within the range of the parameter and false otherwise."
  [keyword type-predicate]
  (let [param (get-in parameter-meta-data [keyword :val-constraint])]
    (s/and type-predicate
           #(<= (first param) % (second param)))))

(s/def ::temperature (get-validate-range-predicate :temperature float?))
(s/def ::top_k (get-validate-range-predicate :top_k int?))
(s/def ::top_p (get-validate-range-predicate :top_p float?))
(s/def ::typical_p (get-validate-range-predicate :typical_p float?))
(s/def ::epsilon_cutoff (get-validate-range-predicate :epsilon_cutoff float?))
(s/def ::eta_cutoff (get-validate-range-predicate :eta_cutoff float?))
(s/def ::diversity_penalty (get-validate-range-predicate :diversity_penalty float?))
(s/def ::repetition_penalty (get-validate-range-predicate :repetition_penalty float?))
(s/def ::encoder_repetition_penalty (get-validate-range-predicate :encoder_repetition_penalty float?))
(s/def ::length_penalty (get-validate-range-predicate :length_penalty float?))
(s/def ::num_beams (get-validate-range-predicate :num_beams int?))
(s/def ::penalty_alpha (get-validate-range-predicate :penalty_alpha float?))
(s/def ::max_new_tokens int?)
(s/def ::min_length int?)
(s/def ::min_new_tokens int?)
(s/def ::early_stopping boolean?)
(s/def ::max_time float?)
(s/def ::num_beam_groups int?)
(s/def ::use_cache boolean?)
(s/def ::model-params-db (s/keys :req-un [::temperature ::top_k ::top_p ::typical_p
                                          ::epsilon_cutoff ::eta_cutoff ::diversity_penalty
                                          ::repetition_penalty ::encoder_repetition_penalty
                                          ::length_penalty ::max_new_tokens ::min_length
                                          ::min_new_tokens ::early_stopping ::max_time
                                          ::num_beams ::num_beam_groups ::penalty_alpha
                                          ::use_cache]))

(defn initial-db
  "Very arbitrary. Should be fetched from metadata endpoint eventually." ;; TODO: fetch from metadata endpoint
  []
  (array-map :temperature 0.9
             :top_k 50
             :top_p 0.4
             :typical_p 1.0
             :epsilon_cutoff 0.0
             :eta_cutoff 0.0
             :diversity_penalty 0.0
             :repetition_penalty 1.0
             :encoder_repetition_penalty 1.0
             :length_penalty 1.0
             :max_new_tokens 2048
             :min_length 0
             :min_new_tokens 0
             :early_stopping false
             :max_time 0.0
             :num_beams 1
             :num_beam_groups 1
             :penalty_alpha 0.0
             :use_cache true))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;           Rich Comments            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(comment
  ;; check if initial-db is valid
  (s/valid? ::model-params-db (initial-db)))
