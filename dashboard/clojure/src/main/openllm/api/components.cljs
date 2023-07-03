(ns openllm.api.components
  (:require [re-frame.core :as rf]
            [reagent.core :as r]))

(defn file-upload
  "The file upload reagent custom component."
  [{:keys [callback-event class]
    :or {class "w-full mt-3 shadow-sm rounded-md focus:z-10 file:bg-transparent file:border-0 file:bg-blue-800 file:mr-4 file:py-2 file:px-4 bg-blue-600 text-white hover:bg-blue-700 file:text-white"}}]
  (let [file-reader (js/FileReader.)]
    (r/create-class
     {:component-did-mount
      (fn [_]
        (.addEventListener file-reader "load"
                           (fn [evt]
                             (let [content (-> evt .-target .-result)]
                               (rf/dispatch [callback-event content])))))
      :render
      (fn []
        [:input {:type "file"
                 :class class
                 :on-change (fn [evt]
                              (let [file (-> evt .-target .-files (aget 0))]
                                (.readAsText file-reader file)))}])})))
