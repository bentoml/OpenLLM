(ns openllm.events
    (:require [cljs.spec.alpha :as s]
              [openllm.db :as db]
              [re-frame.core :refer [after reg-event-db]]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;              Functions             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn check-and-throw
  "Throws an exception if `db` doesn't match the Spec `a-spec`. Acts as a helper
   for our spec checking interceptor."
  [a-spec db]
  (when-not (s/valid? a-spec db)
    (throw (ex-info (str "spec check failed: " (s/explain-str a-spec db)) {}))))

(def check-spec-interceptor
  "The interceptor we will use to check the app-db after each event handler runs.
   It will check that the app-db is valid against the spec `::db`."
  (after (partial check-and-throw :openllm.db/db)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;               Events               ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(reg-event-db
 :initialise-db
 [check-spec-interceptor] ;; why? to force people to update the spec :D
 (fn [_ _]
   db/default-db))

(reg-event-db
 :set-screen-id
 [check-spec-interceptor]
 (fn [db [_ new-screen-id]]
   (assoc db :screen-id new-screen-id)))
