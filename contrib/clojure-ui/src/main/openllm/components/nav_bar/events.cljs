(ns openllm.components.nav-bar.events
  (:require [openllm.api.log4cljs.core :refer [log]]
            [clojure.string :as str]
            [re-frame.core :refer [reg-event-fx]]))

;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;              Functions             ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(defn- start-download!
  "Starts the download of a file."
  [file-name content]
  (let [blob (js/Blob. #js [content] #js {:type "text/plain"})
        link (js/document.createElement "a")]
    (set! (.-href link) (js/window.URL.createObjectURL blob))
    (set! (.-download link) file-name)
    (set! (.-target link) "_blank")
    (.click link)))

(defn- build-playground-export-contents
  "Returns the contents of the playground export file."
  [prompt response]
  (str "Prompt: " prompt
       "\n\n\n"
       "Response: " response))

(defn- build-chat-export-contents
  "Returns the contents of the chat export file."
  [chat-history]
  (let [mapping-fn (fn [current-entry]
                     (str/join ": "
                               [(name (:user current-entry))
                                (:text current-entry)]))]
    (str/join "\n"
              (map mapping-fn
                   chat-history))))


;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
;;               Events               ;;
;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;;
(reg-event-fx
 ::export-button-clicked
 (fn [cofx _]
   {:dispatch-sync true
    :fx (let [active-screen (get-in cofx [:db :screen-id])]
          (condp = active-screen
            :playground
            (let [input-value (get-in cofx [:db :playground-input-value])
                  response-value (get-in cofx [:db :playground-last-response])]
            (start-download! "export-playground.txt"
                             (build-playground-export-contents input-value
                                                               response-value)))
            :chat
            (let [chat-history (get-in cofx [:db :chat-history])]
              (start-download! "export-chat.txt"
                               (build-chat-export-contents chat-history)))
            ;default
            (log :error "Export button clicked in unknown screen.")))}))
