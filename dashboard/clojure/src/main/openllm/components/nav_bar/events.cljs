(ns openllm.components.nav-bar.events
  (:require [openllm.api.log4cljs.core :refer [log]]
            [clojure.string :as str]
            [re-frame.core :refer [reg-event-fx]]))

(defn start-download!
  "Starts the download of a file."
  [file-name content]
  (let [blob (js/Blob. #js [content] #js {:type "text/plain"})
        link (js/document.createElement "a")]
    (set! (.-href link) (js/window.URL.createObjectURL blob))
    (set! (.-download link) file-name)
    (set! (.-target link) "_blank")
    (.click link)))

(reg-event-fx
 ::export-button-clicked
 (fn [cofx _]
   {:dispatch-sync true
    :fx (let [textareas (js/document.querySelectorAll "textarea")
              active-screen (get-in cofx [:db :screen-id])] 
          (condp = active-screen
            :playground
            (start-download! "export-playground.txt"
                             (str "Prompt: " (str/join "\n\n\nResponse: "
                                                       (map #(.-value %) textareas))))
            :chat
            (start-download! "export-chat.txt"
                             (str/join "\n" (map #(str/join ": " [(name (:user %)) (:text %)])
                                                 (get-in cofx [:db :chat-history]))))
            ;default
            (log :error "Export button clicked in unknown screen.")))}))
