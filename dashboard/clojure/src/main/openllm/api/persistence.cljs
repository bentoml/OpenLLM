(ns openllm.api.persistence
  (:require [openllm.api.indexed-db.core :as idb]
            [re-frame.core :refer [reg-event-db] :as rf]))

(def idb-info {:db-name "OpenLLM_clj_GutZuFusss"
               :db-version 1})
(def idb-table-info
  {:name "chat-history"
   :index [{:name "user" :unique false}]})


;; SUBSCRIPTIONS


;; EVENTS
(reg-event-db
 ::set-indexed-db
 (fn [db [_ idb]]
   (assoc db :idb idb)))

;; FUNCTIONS
(defn on-db-initialized
  "Passed as the callback function to `idb/initialize!` to set the
   :idb key in the app-db."
  [idb]
  (rf/dispatch-sync [::set-indexed-db idb]))

(defn init-idb
  "Initializes the IndexedDB database and creates the object store
   if it does not exist."
  []
  (idb/initialize! on-db-initialized idb-info idb-table-info))
