(ns openllm.api.events
  (:require [ajax.core :as ajax]
            [re-frame.core :refer [reg-event-fx]]))

(def api-base-url "http://localhost:3000")

(reg-event-fx
 ::v1-generate
 []
 (fn [_ [_ prompt llm-config & {:keys [on-success on-failure]}]]
   {:http-xhrio {:method :post
                 :uri (str api-base-url "/v1/generate")
                 :params {:prompt prompt
                          :llm_config llm-config}
                 :format          (ajax/json-request-format)
                 :response-format (ajax/json-response-format {:keywords? true})
                 :on-success      on-success
                 :on-failure      on-failure}}))
