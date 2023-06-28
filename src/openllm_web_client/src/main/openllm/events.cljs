(ns openllm.events
    (:require [openllm.db :refer [default-db]]
              [re-frame.core :refer [reg-event-db reg-event-fx after]]
              [ajax.core :as ajax]
              [cljs.spec.alpha :as s]))

(def api-base-url "http://localhost:3000")

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
   default-db))

(reg-event-db
 :set-screen-id
 [check-spec-interceptor]
 (fn [db [_ new-screen-id]]
   (assoc db :screen-id new-screen-id)))

(reg-event-db
 :toggle-model-dropdown
 [check-spec-interceptor]
 (fn [db _]
  (assoc db :model-dropdown-active? (not (:model-dropdown-active? db)))))

(reg-event-db
 :set-chat-input-value
 [check-spec-interceptor]
 (fn [db [_ new-value]]
   (assoc db :chat-input-value new-value)))

(reg-event-fx
 :send-prompt
 []
 (fn [_ [_ prompt llm-config]]
   {:http-xhrio {:method :post
                 :uri (str api-base-url "/v1/generate")
                 :params {:prompt prompt
                          :llm_config llm-config}
                 :format          (ajax/json-request-format)
                 :response-format (ajax/json-response-format {:keywords? true})
                 :on-success      [:send-prompt-success]
                 :on-failure      [:send-prompt-failure]}})) ;; TODO: register handler

(reg-event-db
 :add-to-chat-history
 [check-spec-interceptor]
 (fn [db [_ user text]]
   (assoc db :chat-history (conj (:chat-history db) {:user user
                                                     :text text}))))

(reg-event-fx
 :send-prompt-success
 []
 (fn [_ [_ response]]
   {:dispatch [:add-to-chat-history :model (first (:responses response))]}))

(reg-event-fx
 :on-send-button-click 
 []
 (fn [_ [_ prompt llm-config]]
   {:dispatch-n [[:send-prompt prompt llm-config]
                 [:add-to-chat-history :user prompt]
                 [:set-chat-input-value ""]]}))
