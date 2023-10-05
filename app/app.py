#%%
import streamlit as st
import pandas as pd
import lightgbm as lgb
#%%
def main():
    st.title("RUL Prediction")
    st.write("")
    left, right = st.columns([0.7, 0.3])
    
    with left:
        st.columns(4)[1].subheader("Data")
        
        # File upload
        uploaded_file = st.file_uploader("Upload Assets Sensor Data",
                                        type=["csv", "txt"])
        
        if uploaded_file is not None:
            try:
                # Read the CSV file
                df_input = pd.read_csv(uploaded_file,
                    sep=" ",
                    names=['AssetId', 'Runtime', 'Setting1', 'Setting2', 
                           'Setting3', 'Tag1', 'Tag2', 
                           'Tag3', 'Tag4', 'Tag5', 'Tag6', 'Tag7', 
                           'Tag8', 'Tag9', 'Tag10', 'Tag11', 
                           'Tag12', 'Tag13', 'Tag14', 'Tag15', 
                           'Tag16', 'Tag17', 'Tag18', 'Tag19', 
                           'Tag20', 'Tag21'
                           ],
                    usecols=list(range(0,26))
                    )
                
                columns = ['AssetId', 'Runtime', 'Tag2', 'Tag3', 'Tag4', 
                           'Tag7', 'Tag8', 'Tag9', 'Tag11', 'Tag12', 'Tag13', 
                           'Tag14', 'Tag15', 'Tag17', 'Tag20', 'Tag21'
                           ]
                
                df = df_input[columns]
                
                max_runtime_indices = df.groupby('AssetId')['Runtime'].idxmax()
                df = df.loc[max_runtime_indices].reset_index(drop=True)
                
            except:
                print('Invalid file format.')
            
        with right:
        
            survival_model = lgb.Booster(model_file='model.pkl')
            
            # Make predictions on test data
            y_pred = survival_model.predict(df, 
                                            num_iteration=survival_model.best_iteration
                                            )
            
            results = pd.concat([df['AssetId'], pd.Series(y_pred, name='RUL')], 
                                axis=1)
            
            # Display the DataFrame
            st.columns(4)[1].subheader("Results")
            
            st.dataframe(results, hide_index=True, use_container_width=True)
            
            # Download link
            csv = results.to_csv(index=False)
            st.download_button("Download Results", data=csv, 
                                file_name="results.csv", 
                                mime="text/csv",
                                use_container_width=True)

if __name__ == "__main__":
    main()