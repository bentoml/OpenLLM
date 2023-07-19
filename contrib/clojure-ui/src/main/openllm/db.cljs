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

;; Below is is the root `clojure.spec` specification for the value in app-db.
;; It basically works like a like a schema.
;; See: http://clojure.org/guides/spec
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
(s/def ::db  (s/keys :req-un [::components-db/components-db
                              ::screen-id]))

(def default-db
  "What gets put into app-db by default.
   See 'core.cljs' for `(dispatch-sync [:initialise-db])` and 'events.cljs'
   for the registration of `:initialise-db` effect handler."
  {:components-db (components-db/initial-db)
   :screen-id :playground})


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
