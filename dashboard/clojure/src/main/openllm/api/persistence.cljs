(ns openllm.api.persistence
  (:require [openllm.api.indexed-db.core :as idb]
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
  (rf/dispatch-sync [::set-indexed-db idb])
  (rf/dispatch [::sync-chat-history]))

(defn init-idb
  "Initializes the IndexedDB database and creates the object store
   if it does not exist.
   
   This function notably registers the `on-db-initialized` function
   as a callback function to be called when the IndexedDB database
   is initialized."
  []
  (idb/initialize! idb-info idb-table-info on-db-initialized))

(defn idb-chat-history->clean-chat-history
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
 (fn [cofx [_ user message]]
   (let [indexed-db (:idb cofx)]
     (idb/os-add! {:db indexed-db :os-name (:name idb-table-info)}
                  {:user user :text message}))))

;; This event will override the chat history in the app-db with the data from
;; the IndexedDB database. It will be dispatched as a callback function to
;; `idb/os-get-all`, which is called in the `::sync-chat-history` event from
;; this namespace.
(rf/reg-event-db
 ::set-chat-history-app-db
 (fn [db [_ chat-history]]
   (let [clean-chat-history (idb-chat-history->clean-chat-history chat-history)]
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


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;           Rich Comments            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(comment
  ;; add a chat message to the database
  (rf/dispatch [::add-to-indexed-db-history :model "hello"])

  ;; cleaning chat history demo
  (def chat-history [{:id 1 :user "model" :text "hi"}
                     {:id 2 :user "user" :text "hey"}])
  (idb-chat-history->clean-chat-history chat-history) ;; => [{:user :model, :text "hi"} {:user :user, :text "hey"}]

  ;; set the chat history in the app-db to the data from the indexed-db database
  (rf/dispatch [::sync-chat-history]))
