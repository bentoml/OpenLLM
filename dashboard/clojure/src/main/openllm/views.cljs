(ns openllm.views
  (:require [re-frame.core :as rf]
            [openllm.components.nav-bar.views :as nav-bar-views]
            [openllm.components.side-bar.views :as side-bar-views]
            [openllm.components.playground.views :as playground-views]
            [openllm.components.chat.views :as chat-views]
            [openllm.components.chat.events :as chat-events]))

(defn tabs
  "The tabs at the top of the screen. The selected tab decides, which
   content is rendered in the central area of the screen."
  [screen-id]
  [:div {:class "mt-4 grid grid-cols-3 bg-white rounded-lg shadow divide-x divide-gray-300"}
   [:button {:class (if (= screen-id :playground) "bg-gray-700 text-white font-bold py-2 px-4 rounded-l-lg"
                                                  "bg-white shadow divide-x divide-gray-300 rounded-l-lg hover:bg-gray-100")
             :on-click #(rf/dispatch [:set-screen-id :playground])} "Playground"]
   [:button {:class (if (= screen-id :chat) "bg-gray-700 text-white font-bold py-2 px-4"
                                            "bg-white shadow divide-x divide-gray-300 hover:bg-gray-100")
             :on-click #(do (rf/dispatch-sync [:set-screen-id :chat])
                            (rf/dispatch [::chat-events/auto-scroll]))} "Chat"]
   [:button {:class (if (= screen-id :apis) "bg-gray-700 text-white font-bold py-2 px-4 rounded-r-lg"
                                            "bg-white shadow divide-x divide-gray-300 rounded-r-lg hover:bg-gray-100")
             :on-click #(rf/dispatch [:set-screen-id :apis])} "APIs"]])

(defn tab-content
  "The content of the currently active tab. Essentially, this is the
   main router of the app, deciding what content should be rendered
   based on the current screen-id."
  [screen-id]
  (case screen-id
    :playground [playground-views/playground-tab-contents]
    :chat [chat-views/chat-tab-contents]))

(defn dashboard
  "The main dashboard component, which is rendered into the DOM. Called
   directly by the `app` function residing inside the `app` namespace,
   which is the main component and rendered by the root rendering function
   of this application. Called is putting it mildly, as this function is
   basically wrapped by the `app` function, thus this component contains
   everything that is rendered into the DOM as it's children."
  []
  (let [screen-id (rf/subscribe [:screen-id])]
    (fn []
      [:div {:class "h-screen flex bg-white"}
       [:div {:class "flex flex-col flex-1 w-screen"}
        [:main {:class "flex-1 relative z-0 overflow-hidden focus:outline-none" :tabIndex "0"}
         [nav-bar-views/nav-bar]
         [:div {:class "mt-2 pr-0.5 pl-4 w-full h-full"}
          [tab-content @screen-id]]]]
       [side-bar-views/side-bar]])))
