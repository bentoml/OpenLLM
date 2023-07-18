(ns openllm.db
  "This namespace acts as app-db for the application. This namespace
   and the backing data structure should be the only part of the application
   that has state and is mutable.
   Other parts of the application should stay as pure as possible.

   Do not directly interact with this namespace! Use the 'subs'
   (subscriptions) and events namespaces for this.
   The subscription namespaces may read the respective data that is of
   interest to them.
   The event namespaces may be used to change (thus WRITE to the) app-db."
  (:require [openllm.components.db :as components-db]
            [cljs.spec.alpha :as s]))

;; Below is is a clojure.spec specification for the value in app-db. It is
;; like a Schema. See: http://clojure.org/guides/spec
;;
;; The value in app-db should always match this spec. Only event handlers
;; can change the value in app-db so, after each event handler
;; has run, we re-check app-db for correctness (compliance with the Schema).
;;
;; How is this done? Look in events.cljs and you'll notice that all handlers
;; have an "after" interceptor which does the spec re-check.
;; None of this is strictly necessary. It could be omitted. But we (the
;; re-frame people) find it good practice.

(s/def ::screen-id keyword?)
(s/def ::side-bar-open? boolean?)
(s/def ::modal-open? (s/keys :req-un [::playground boolean?]))
(s/def ::selected-model (s/keys :req-un [::model-type keyword?
                                         ::model-id string?]))

(s/def ::playground-input-value string?)
(s/def ::playground-last-response string?)

;; ########################## MODEL CONFIG ##########################
(def parameter-min-max
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

(defn get-validate-range-predicate [keyword type-predicate]
  (let [param (keyword parameter-min-max)]
    (s/and type-predicate
           #(<= (first param) % (second param)))))

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
;; ####################### MODEL CONFIG  END ########################

;; ########################### AGGREGATE ############################
(s/def ::db  (s/keys :req-un [::components-db/components-db
                              ::screen-id
                              ::side-bar-open?
                              ::modal-open?
                              ::selected-model
                              ::playground-input-value
                              ::playground-last-response
                              ::model-config]))
;; ######################## AGGREGATE END ###########################

(def standard-llm-config
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

(def default-db
  "What gets put into app-db by default.
   See 'core.cljs' for `(dispatch-sync [:initialise-db])` and 'events.cljs'
   for the registration of `:initialise-db` effect handler."
  {:components-db components-db/initial-db
   :screen-id :playground
   :side-bar-open? true
   :modal-open? {:playground false}
   :selected-model {:model-type :chatglm
                    :model-id "google/chatglm-6b"}
   :playground-input-value ""
   :playground-last-response ""
   :model-config standard-llm-config})


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;           Rich Comments            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(comment
  ;; some examples on namespaced keywords
  (= :components-db/components-db :openllm.components.db/components-db) ;; => false
  (= ::components-db/components-db :openllm.components.db/components-db) ;; => true
  (= ::components-db/components-db ::openllm.components.db/components-db) ;; => true
  (= :components-db :openllm.components.db) ;; => false


  ;; check if default db complies with spec
  (s/valid? ::db default-db) ;; => true


  ;; check if manipulated db (no screen-id key) complies with spec
  (s/valid? ::db (dissoc default-db :screen-id)) ;; => false


  ;; reset app-db to default-db
  (do (require '[re-frame.core :as rf])
      (rf/dispatch-sync [:initialise-db])))
