(ns openllm.db
  "This namespace acts as the root app-db for the application. The `db` namespaces
   define the structure of the `app-db`. They each define a schema for the
   respective branch of the `app-db`. The `db` namespaces should only be used by
   the `events` and `subs` namespaces.
   The `clojure.spec` schema is checked against the `app-db` after each event
   handler has run. This is done by the `events` namespaces with the help of the
   `check-spec-interceptor` from the `openllm.events` namespace.
   
   Please note that each `db` namespace has an `initial-db` function which
   returns the initial value for each branch of the `app-db`.

   Furthermore be aware that the `app-db` is immutable. This means that you cannot
   change the `app-db` directly. Instead you have to dispatch an event which will
   then change the `app-db` as a whole."
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
  "What gets put into app-db by default. This is the very root of the `app-db`.
   See `core.cljs` for `(dispatch-sync [:initialise-db])` and 'events.cljs'
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
