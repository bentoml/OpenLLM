(ns openllm.views
  (:require [re-frame.core :as rf]
            [openllm.db :as db]
            [clojure.string :as str])) ;; TODO: remove this. just for the standard llm-config for now

(defn second-page
  []
  [:div {:class "min-h-screen bg-gray-50 flex flex-col justify-center py-12 sm:px-6 lg:px-8"}
   [:div {:class "mt-8 sm:mx-auto sm:w-full sm:max-w-md"}
    [:div {:class "bg-white py-8 px-4 shadow sm:rounded-lg sm:px-10"}
     [:h2 {:class "mt-6 text-center text-3xl font-extrabold text-gray-900"} "Another page"]]]])

(defn openllm-tag
  "The 'OpenLLM' tag in the very top left corner of the screen."
  []
  [:div {:class "flex items-center flex-shrink-0 px-6"}
   [:img {:class "h-11 w-auto" :src "./static/logo-light.svg" :alt "LOGO"}]
   [:span {:class "text-3xl font-bold text-gray-900 ml-2"} "OpenLLM"]])

(defn sidebar-group-headline
  "The headlines for the different groups in the sidebar are rendered using this component."
  [headline]
  [:h3 {:class "px-4 text-xs font-semibold text-gray-500 uppercase tracking-wider" :id (str "sidebar-headline-" headline)} headline])

(defn parameter-slider-with-input
  "Renders a slider with an input field next to it."
  [name value]
  (let [min-max (name db/parameter-min-max)
        num-type? (or (str/includes? (str name) "num") (= name ::db/top_k))
        on-change #(rf/dispatch [:set-model-config-parameter name (if num-type?
                                                                    (parse-long (.. % -target -value))
                                                                    (parse-double (.. % -target -value)))])]
    [:div {:class "flex flex-row items-center"}
     [:span {:class "mr-2 text-xs text-gray-500"} (str (first min-max))]
     [:input {:type "range"
              :min (first min-max)
              :max (second min-max)
              :step (if num-type? 1 0.01)
              :value value
              :class "w-28 h-2 bg-gray-300 accent-pink-600"
              :on-change on-change}]
     [:span {:class "ml-2 text-xs text-gray-500"} (str (second min-max))]
     [:input {:type "number"
              :class "w-16 px-1 py-1 text-xs text-gray-700 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-pink-500 focus:border-pink-500 sm:text-sm ml-auto"
              :step (if num-type? 1 0.01)
              :value value
              :on-change on-change}]]))

(defn parameter-checkbox
  "Renders a checkbox."
  [name value]
  [:input {:type "checkbox"
           :checked value
           :class "h-4 w-4 text-pink-600 focus:ring-pink-500 border-gray-300 rounded"
           :on-change #(rf/dispatch [:set-model-config-parameter name (boolean (.. % -target -checked))])}])

(defn parameter-list-entry-value
  [name value]
  (cond
    (contains? db/parameter-min-max name) [parameter-slider-with-input name value]
    (boolean? value) [parameter-checkbox]
    :else
    [:input {:type "number"
             :class "px-1 py-1 text-xs text-gray-700 border border-gray-300 rounded-md shadow-sm focus:outline-none focus:ring-pink-500 focus:border-pink-500 sm:text-sm w-full"
             :value value
             :on-change #(rf/dispatch [:set-model-config-parameter name (int (.. % -target -value))])}]))

(defn parameter-list-entry
  "Renders a single parameter in the sidebar's parameter list."
  [[parameter-name parameter-value]]
  [:div {:class "flex flex-col px-3 py-2 text-sm font-medium text-gray-700"
         :key (str parameter-name)}
   [:span {:class "text-gray-500"} parameter-name]
   [:div {:class "mt-1"}
    [parameter-list-entry-value parameter-name parameter-value]]])

(defn parameter-list
  "Renders the parameters in the sidebar."
  []
  (let [model-config (rf/subscribe [:model-config])]
    (fn parameter-list
      []
      [:div
       (map parameter-list-entry @model-config)])))

(defn status-display
  "Displays the current service status at the bottom of the sidebar."
  [status-good?]
  (let [status-color (if status-good? "bg-green-500" "bg-red-500")
        status-text (if status-good? "Operational" "Degraded Performance")]
    [:div {:class "px-3 mt-6"}
     [:div {:class "mt-8"}
      [sidebar-group-headline "Service Status"]
      [:div {:class "space-y-1" :role "group" :aria-labelledby "service-status-headline"}
       [:a {:href "#" :class "group flex items-center px-3 py-1 text-sm font-medium text-gray-700 rounded-md"}
        [:span {:class (str "w-2.5 h-2.5 mr-4 rounded-full " status-color) :aria-hidden "true"}]
        [:span {:class "truncate"} status-text]]]]]))

(defn side-bar
  "The render function of the toolbar on the very left of the screen"
  []
  [:div {:class "flex flex-col w-80 border-r border-gray-200 pt-5 pb-4 bg-gray-200"} ;; sidebar div + background
   [openllm-tag]
   [:hr {:class "my-5 border-1 border-black"}]
   [sidebar-group-headline "Parameters"]
   [:div {:class "my-4 h-0 flex-1 flex flex-col overflow-y-auto"}
    [:div {:class "px-3 mt-8 relative inline-block text-left"}
     [parameter-list]]]
   [status-display true]])

(defn chat-controls
  "The chat input field and the send button."
  []
  (let [chat-input-sub (rf/subscribe [:chat-input-value])
        on-change #(rf/dispatch [:set-chat-input-value (.. % -target -value)])
        on-send-click #(rf/dispatch [:on-send-button-click @chat-input-sub db/standard-llm-config])]
    (fn chat-controls
      []
      [:div {:class "fixed bottom-0 px-4 py-2 mt-6 w-10/12"}
       [:form {:class "flex items-center justify-between"
               :on-submit #(do % (on-send-click)
                               (.preventDefault %))}
        [:textarea {:class "py-1 w-[calc(100%_-_80px)] appearance-none block border border-gray-300 rounded-md shadow-sm placeholder-gray-400 focus:outline-none focus:ring-pink-500 focus:border-pink-500 sm:text-sm"
                    :type "text" :placeholder "Type your prompt..."
                    :value @chat-input-sub
                    :on-change on-change
                    :id "chat-input"
                    :auto-complete "off"
                    :auto-correct "off"}]
        [:button {:class "ml-2 px-4 py-2 text-white bg-blue-600 rounded-md hover:bg-blue-700 focus:outline-none"
                  :on-click on-send-click
                  :type "button"} "Send"]]])))

(defn chat-history
  "The chat history."
  []
  (let [history (rf/subscribe [:chat-history])]
    (fn chat-history []
      (into [:div {:class "flex-1 overflow-auto mt-6"}]
            (map (fn [{:keys [user text]}]
                   (let [diplay-user (if (= user :model) "Model" "You")
                         color (if (= user :model) "bg-gray-200" "bg-blue-200")]
                     [:div {:class (str "p-2 rounded-lg mb-2 " color)}
                      [:h3 {:class "font-bold text-lg"} diplay-user]
                      [:p {:class "text-gray-700"} text]]))
                 @history)))))

(defn tabs
  "The tabs at the top of the screen."
  [screen-id]
  [:div {:class "mt-3 grid grid-cols-3 bg-white rounded-lg shadow divide-x divide-gray-200"}
   [:button {:class (when (= screen-id :playground) "bg-pink-600 hover:bg-red-pink-700 text-white font-bold py-2 px-4 rounded-l")
             :on-click #(rf/dispatch [:set-screen-id :playground])} "Playground"]
   [:button {:class (when (= screen-id :chat) "bg-pink-600 hover:bg-red-pink-700 text-white font-bold py-2 px-4 rounded-l")
             :on-click #(rf/dispatch [:set-screen-id :chat])} "Chat"]
   [:button {:class (when (= screen-id :apis) "bg-pink-600 hover:bg-red-pink-700 text-white font-bold py-2 px-4 rounded-l")
             :on-click #(rf/dispatch [:set-screen-id :apis])} "APIs"]])

(defn dashboard
  []
  (let [screen-id (rf/subscribe [:screen-id])]
    (fn []
      [:div {:class "h-screen flex overflow-hidden bg-white"}
       [:div {:class "flex flex-col w-0 flex-1 overflow-hidden"}
        [:main {:class "flex-1 relative z-0 overflow-y-auto focus:outline-none" :tabIndex "0"}
         [:div {:class "px-4 mt-6 sm:px-6 lg:px-8"}
          [:h2 {:class "text-gray-500 text-xs font-medium uppercase tracking-wide"} "Dashboard"]
          ;; 3 tabs: Playground, Chat, APIs 
          [tabs @screen-id]
          (case @screen-id
            :playground [:div]
            :chat [chat-history]
            :apis [:div])]]]
       (when (= @screen-id :chat) [chat-controls])
       [:div {:class "hidden lg:flex lg:flex-shrink-0"}
        [side-bar]]])))
