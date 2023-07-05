(ns openllm.components.apis.data
  (:require [openllm.api.http :as http]))

(def endpoints-data
  "This is the static data for rendering the endpoints."
  [{:id :v1-generate
    :name "/v1/generate"
    :event [::http/v1-generate-raw]}

   {:id :v1-metadata
    :name "/v1/metadata"
    :event [::http/v1-metadata]}])
