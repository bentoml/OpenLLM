(ns openllm.api.persistence
  (:require [openllm.api.indexed-db.core :as idb]
            [openllm.api.log4cljs.core :refer [log]]
            [re-frame.core :as rf]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;              Constants             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(def idb-info {:db-name "OpenLLM_clj_GutZuFusss"
               :db-version 1})

(def idb-table-info
  {:name "chat-history"
   :index [{:name "user" :unique false}]})


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;              Functions             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn on-db-initialized
  "Passed as the callback function to `idb/initialize!` to set the
   :idb key in the app-db.

   This function also dispatches the `::sync-chat-history` event to
   populate the chat history in the app-db with the data from the
   IndexedDB database."
  [idb]
  (log :debug "IndexedDB database initialized")
  (rf/dispatch-sync [::set-indexed-db idb])
  (rf/dispatch [::sync-chat-history]))

(defn init-idb
  "Initializes the IndexedDB database and creates the object store
   if it does not exist.

   This function notably registers the `on-db-initialized` function
   as a callback function to be called when the IndexedDB database
   is initialized."
  []
  (log :debug "Initializing IndexedDB database...")
  (idb/initialize! idb-info idb-table-info on-db-initialized))

(defn chat-history->sanitized
  "Takes the chat history from the IndexedDB database and cleans it
   up to be used in the app-db.

   First, the `:id` key is removed from each message, and then the
   values belonging to the `:user` keys are converted to a keyword.
   Finally, the chat history is converted to a vector."
  [chat-history]
  (let [mapping-fn (fn [message] (-> message
                                     (dissoc , :id)
                                     (assoc , :user (keyword (:user message)))))]
    (vec
     (map mapping-fn chat-history))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;              Coeffects             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(rf/reg-cofx
 ::indexed-db
 (fn [cofx _]
   (assoc cofx :idb (get-in cofx [:db :indexed-db]))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;               Events               ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(rf/reg-event-db
 ::set-indexed-db
 (fn [db [_ idb]]
   (assoc db :indexed-db idb)))

;; Adds a chat message to the IndexedDB database. Is dispatched when the user
;; sends or receives a message in the chat.
(rf/reg-event-fx
 ::add-to-indexed-db-history
 [(rf/inject-cofx ::indexed-db)]
 (fn [cofx [_ timestamp user message]]
   (let [indexed-db (:idb cofx)]
     (idb/os-add! {:db indexed-db :os-name (:name idb-table-info)}
                  {:user user :text message
                   :timestamp timestamp}))))

;; This event will override the chat history in the app-db with the data from
;; the IndexedDB database. It will be dispatched as a callback function to
;; `idb/os-get-all`, which is called in the `::sync-chat-history` event from
;; this namespace.
(rf/reg-event-db
 ::set-chat-history-app-db
 (fn [db [_ chat-history]]
   (let [clean-chat-history (chat-history->sanitized chat-history)]
     (log :debug "Synchronized chat history with IndexedDB database, loaded"
          (count clean-chat-history) "messages.")
     (assoc db :chat-history clean-chat-history))))

;; Will be dispatched when the IndexedDB database is initialized, and will
;; populate the chat history in the app-db with the data from the IndexedDB
;; database.
;; Passes the `dispatch` function for the `::set-chat-history-app-db` event
;; to the callback function of `idb/os-get-all`.
(rf/reg-event-fx
 ::sync-chat-history
 [(rf/inject-cofx ::indexed-db)]
 (fn [cofx [_]]
   (let [indexed-db (:idb cofx)
         callback-fn (fn [result]
                       (rf/dispatch [::set-chat-history-app-db (js->clj result :keywordize-keys true)]))]
     (idb/os-get-all {:db indexed-db :os-name (:name idb-table-info)} callback-fn))))

(rf/reg-event-fx
 ::clear-chat-history
 [(rf/inject-cofx ::indexed-db)]
 (fn [cofx [_]]
   (let [indexed-db (:idb cofx)]
     (idb/wipe-object-store! {:db indexed-db :os-name (:name idb-table-info)}))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;           Rich Comments            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(comment
  ;; add a chat message to the database
  (rf/dispatch [::add-to-indexed-db-history :model "hello"])

  ;; sanitizing chat history demo
  (def chat-history [{:id 1 :user "model" :text "hi"}
                     {:id 2 :user "user" :text "hey"}])
  (chat-history->sanitized chat-history) ;; => [{:user :model, :text "hi"} {:user :user, :text "hey"}]

  ;; set the chat history in the app-db to the data from the indexed-db database
  (rf/dispatch [::sync-chat-history]))
