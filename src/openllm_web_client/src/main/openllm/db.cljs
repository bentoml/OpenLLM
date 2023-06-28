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
  (:require [cljs.spec.alpha :as s]))

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
(s/def ::model-dropdown-active? boolean?)
(s/def ::chat-input-value string?)
(s/def ::chat-history (s/coll-of (s/keys :req-un [::user ::text]) :kind vector?))
(s/def ::db (s/keys :req-un [::screen-id
                             ::model-dropdown-active?
                             ::chat-input-value
                             ::chat-history]))

(def default-db
  "What gets put into app-db by default.
   See 'core.cljs' for `(dispatch-sync [:initialise-db])` and 'events.cljs'
   for the registration of `:initialise-db` handler)"
  {:screen-id :main
   :model-dropdown-active? false
   :chat-input-value ""
   :chat-history []})

(def standard-llm-config
  "Very arbitrary. Review this please." ;; TODO
  {:max_new_tokens 2048
   :min_length 0
   :early_stopping false
   :num_beams 1
   :num_beam_groups 1
   :use_cache true
   :temperature 0.9
   :top_k 50
   :top_p 0.4
   :typical_p 1
   :epsilon_cutoff 0
   :eta_cutoff 0
   :diversity_penalty 0
   :repetition_penalty 1
   :encoder_repetition_penalty 1
   :length_penalty 1
   :no_repeat_ngram_size 0
   :renormalize_logits false
   :remove_invalid_values false
   :num_return_sequences 1
   :output_attentions false
   :output_hidden_states false
   :output_scores false
   :encoder_no_repeat_ngram_size 0})
