(ns openllm.views
  (:require [re-frame.core :as rf]
            [openllm.components.side-bar.views :as side-bar-views]
            [openllm.components.playground.views :as playground-views]
            [openllm.components.chat.views :as chat-views])) 

(defn tabs
  "The tabs at the top of the screen."
  [screen-id]
  [:div {:class "mt-4 grid grid-cols-3 bg-white rounded-lg shadow divide-x divide-gray-300"}
   [:button {:class (if (= screen-id :playground) "bg-pink-700 text-white font-bold py-2 px-4 rounded-l-lg" 
                                                  "bg-white shadow divide-x divide-gray-300 rounded-l-lg hover:bg-gray-100")
             :on-click #(rf/dispatch [:set-screen-id :playground])} "Playground"]
   [:button {:class (if (= screen-id :chat) "bg-pink-700 text-white font-bold py-2 px-4"
                                            "bg-white shadow divide-x divide-gray-300 hover:bg-gray-100")
             :on-click #(rf/dispatch [:set-screen-id :chat])} "Chat"]
   [:button {:class (if (= screen-id :apis) "bg-pink-700 text-white font-bold py-2 px-4 rounded-r-lg"
                                            "bg-white shadow divide-x divide-gray-300 rounded-r-lg hover:bg-gray-100")
             :on-click #(rf/dispatch [:set-screen-id :apis])} "APIs"]])

(defn dashboard
  []
  (let [screen-id (rf/subscribe [:screen-id])]
    (fn []
      [:div {:class "h-screen flex bg-white"}
       [:div {:class "flex flex-col w-0 flex-1"}
        [:main {:class "flex-1 relative z-0 overflow-hidden focus:outline-none" :tabIndex "0"}
         [:div {:class "px-4 mt-6 sm:px-6 lg:px-8 w-full h-full"}
          [:h2 {:class "text-gray-500 text-xs font-medium uppercase tracking-wide"} "Dashboard"]
          [tabs @screen-id]
          (case @screen-id
            :playground [playground-views/playground-tab]
            :chat [chat-views/chat-tab]
            :apis [:div])]]]
       [:div {:class "hidden lg:flex lg:flex-shrink-0"}
        [side-bar-views/side-bar]]])))
