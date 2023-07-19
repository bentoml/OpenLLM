(ns openllm.events
    (:require [cljs.spec.alpha :as s]
              [openllm.db :as db]
              [re-frame.core :refer [after reg-cofx reg-event-db reg-event-fx]]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;              Functions             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn check-and-throw
  "Throws an exception if `db` does not match the Spec `a-spec`. Acts as a helper
   for our spec checking interceptor."
  [a-spec db]
  (when-not (s/valid? a-spec db)
    (throw (ex-info (str "spec check failed: " (s/explain-str a-spec db)) {}))))

(def check-spec-interceptor
  "The interceptor we will use to check the app-db after each event handler runs.
   It will check that the app-db is valid against the spec `::db`."
  (after (partial check-and-throw ::db/db)))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;             Coeffects              ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(reg-cofx
 :time-now
 (fn [cofx _]
   (assoc cofx :time-now (js/Date.))))


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

(reg-event-fx
 ::open-link-in-new-tab
 (fn [_ [_ url]] 
   {:fx (js/window.open url "_blank")})) ;; hitchu with da side fx's *new wave uptempo kick*


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;           Rich Comments            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(comment
  #_{:clj-kondo/ignore [:duplicate-require]}
  (require '[re-frame.core :as rf])

  ;; set screen-id to :chat
  (rf/dispatch-sync [:set-screen-id :chat])

  ;; reset app-db to default-db
  (rf/dispatch-sync [:initialise-db]))
