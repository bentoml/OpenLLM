(ns openllm.api.http
  (:require [ajax.core :as ajax]
            [re-frame.core :refer [reg-event-fx]]))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;              Functions             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- get-uri
  "Returns the URI for the given endpoint."
  [api-base-url endpoint]
  (str api-base-url endpoint))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;               Events               ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(reg-event-fx
 ::v1-generate
 []
 (fn [{:keys [db]} [_ prompt llm-config & {:keys [on-success on-failure]}]]
   (let [base-url (get db :api-base-url)]
     {:http-xhrio {:method          :post
                   :uri             (get-uri base-url "/v1/generate")
                   :params          {:prompt prompt
                                     :llm_config llm-config}
                   :format          (ajax/json-request-format)
                   :response-format (ajax/json-response-format {:keywords? true})
                   :on-success      on-success
                   :on-failure      on-failure}})))

(reg-event-fx
 ::v1-metadata
 []
 (fn [{:keys [db]} [_ json & {:keys [on-success on-failure]}]]
   (let [base-url (get db :api-base-url)]
     {:http-xhrio {:method          :post
                   :uri             (get-uri base-url "/v1/metadata")
                   :params          json
                   :format          (ajax/json-request-format)
                   :response-format (ajax/json-response-format {:keywords? true})
                   :on-success      on-success
                   :on-failure      on-failure}})))
