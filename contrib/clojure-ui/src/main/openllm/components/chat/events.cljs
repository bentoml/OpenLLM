(ns openllm.components.chat.events
    (:require [openllm.events :refer [check-spec-interceptor]]
              [openllm.api.http :as api]
              [openllm.api.persistence :as persistence]
              [openllm.api.log4cljs.core :refer [log]]
              [re-frame.core :refer [reg-cofx reg-event-db reg-event-fx inject-cofx]]
              [clojure.string :as str]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;              Coeffects             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(reg-cofx
 ::chat-history-element
 (fn [cofx _]
   (let [element (js/document.getElementById "chat-history-container")]
     (assoc cofx :chat-history-component element))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;               Events               ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(reg-event-db
 ::set-chat-input-value
 [check-spec-interceptor]
 (fn [db [_ new-value]]
   (assoc db :chat-input-value new-value)))

(reg-event-db
 ::add-to-app-db-history
 [check-spec-interceptor]
 (fn [db [_ user text]]
   (assoc db :chat-history (conj (:chat-history db) {:user user
                                                     :text text}))))

;; Puts the received or sent message into the IndexedDB database aswell
;; as the app-db.
(reg-event-fx
 ::add-to-chat-history
 []
 (fn [_ [_ user text]]
   {:dispatch-n [[::add-to-app-db-history user text]
                 [::persistence/add-to-indexed-db-history user text]]}))

(reg-event-fx
 ::send-prompt-success
 []
 (fn [_ [_ response]]
   {:dispatch [::add-to-chat-history :model (first (:responses response))]}))

(reg-event-fx
 ::send-prompt-failure
 []
 (fn [_ [_ e]]
   (log :error "Failed to send prompt" e)
   {:dispatch-later [{:ms 10 :dispatch [::add-to-chat-history :model "Sorry, something went wrong."]}
                     {:ms 20 :dispatch [::auto-scroll]}]}))

(reg-event-fx
 ::on-send-button-click
 []
 (fn [{:keys [db]} [_ prompt llm-config]]
   (when (not (str/blank? prompt))
     {:dispatch-n [[::add-to-chat-history :user (:chat-input-value db)]
                   [::api/v1-generate prompt llm-config {:on-success [::send-prompt-success]
                                                         :on-failure [::send-prompt-failure]}]
                   [::set-chat-input-value ""]]
      :dispatch-later [{:ms 20 :dispatch [::auto-scroll]}]})))

(reg-event-db
 ::toggle-modal
 [check-spec-interceptor]
 (fn [db [_ _]]
   (let [new-value (not (get-in db [:modal-open? :chat]))]
     (assoc-in db [:modal-open? :chat] new-value))))

(reg-event-db
 ::set-prompt-layout
 [check-spec-interceptor]
 (fn [db [_ layout]]
   (assoc db :prompt-layout layout)))

(reg-event-fx
 ::auto-scroll
 [(inject-cofx ::chat-history-element)]
 (fn [cofx _]
   (let [chat-history-element (get cofx :chat-history-component)]
     (set! (.-scrollTop chat-history-element)
           (.-scrollHeight chat-history-element)))
   {}))

(reg-event-db
 ::clear-chat-history
 []
 (fn [db _]
   (assoc db :chat-history [])))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;           Rich Comments            ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(comment
  ;; import re-frame
  #_{:clj-kondo/ignore [:duplicate-require]}
  (require '[re-frame.core :as rf])

  ;; add a chat message to the app-db (makes is appear in the chat history screen)
  (rf/dispatch [::add-to-app-db-history :model "hello"])

  ;; scroll to the bottom
  (rf/dispatch [::auto-scroll])

  ;; clear the chat history
  (rf/dispatch [::clear-chat-history]))
