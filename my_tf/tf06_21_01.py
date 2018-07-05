# 本代码由可视化策略环境自动生成 2018年2月8日 09:11
# 本代码单元只能在可视化模式下编辑。您也可以拷贝代码，粘贴到新建的代码单元或者策略，然后修改。


m3 = M.dl_layer_input.v1(
    shape='50,5',
    batch_shape='',
    dtype='float32',
    sparse=False,
    name=''
)

m13 = M.dl_layer_reshape.v1(
    inputs=m3.data,
    target_shape='50,5,1',
    name=''
)

m14 = M.dl_layer_conv2d.v1(
    inputs=m13.data,
    filters=32,
    kernel_size='3,5',
    strides='1,1',
    padding='valid',
    data_format='channels_last',
    dilation_rate='1,1',
    activation='relu',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='Zeros',
    kernel_regularizer='None',
    kernel_regularizer_l1=0,
    kernel_regularizer_l2=0,
    bias_regularizer='None',
    bias_regularizer_l1=0,
    bias_regularizer_l2=0,
    activity_regularizer='None',
    activity_regularizer_l1=0,
    activity_regularizer_l2=0,
    kernel_constraint='None',
    bias_constraint='None',
    name=''
)

m15 = M.dl_layer_reshape.v1(
    inputs=m14.data,
    target_shape='48,32',
    name=''
)

m4 = M.dl_layer_lstm.v1(
    inputs=m15.data,
    units=32,
    activation='tanh',
    recurrent_activation='hard_sigmoid',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    recurrent_initializer='Orthogonal',
    bias_initializer='Ones',
    unit_forget_bias=True,
    kernel_regularizer='None',
    kernel_regularizer_l1=0,
    kernel_regularizer_l2=0,
    recurrent_regularizer='None',
    recurrent_regularizer_l1=0,
    recurrent_regularizer_l2=0,
    bias_regularizer='None',
    bias_regularizer_l1=0,
    bias_regularizer_l2=0,
    activity_regularizer='None',
    activity_regularizer_l1=0,
    activity_regularizer_l2=0,
    kernel_constraint='None',
    recurrent_constraint='None',
    bias_constraint='None',
    dropout=0,
    recurrent_dropout=0,
    return_sequences=False,
    implementation='0',
    name=''
)

m11 = M.dl_layer_dropout.v1(
    inputs=m4.data,
    rate=0.8,
    noise_shape='',
    name=''
)

m10 = M.dl_layer_dense.v1(
    inputs=m11.data,
    units=32,
    activation='tanh',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='Zeros',
    kernel_regularizer='None',
    kernel_regularizer_l1=0,
    kernel_regularizer_l2=0,
    bias_regularizer='None',
    bias_regularizer_l1=0,
    bias_regularizer_l2=0,
    activity_regularizer='None',
    activity_regularizer_l1=0,
    activity_regularizer_l2=0,
    kernel_constraint='None',
    bias_constraint='None',
    name=''
)

m12 = M.dl_layer_dropout.v1(
    inputs=m10.data,
    rate=0.8,
    noise_shape='',
    name=''
)

m9 = M.dl_layer_dense.v1(
    inputs=m12.data,
    units=1,
    activation='sigmoid',
    use_bias=True,
    kernel_initializer='glorot_uniform',
    bias_initializer='Zeros',
    kernel_regularizer='None',
    kernel_regularizer_l1=0,
    kernel_regularizer_l2=0,
    bias_regularizer='None',
    bias_regularizer_l1=0,
    bias_regularizer_l2=0,
    activity_regularizer='None',
    activity_regularizer_l1=0,
    activity_regularizer_l2=0,
    kernel_constraint='None',
    bias_constraint='None',
    name=''
)

m5 = M.dl_model_init.v1(
    inputs=m3.data,
    outputs=m9.data
)

m8 = M.input_features.v1(
    features="""(close/shift(close,1)-1)*10
(high/shift(high,1)-1)*10
(low/shift(low,1)-1)*10
(open/shift(open,1)-1)*10
(volume/shift(volume,1)-1)*10"""
)

m25 = M.instruments.v2(
    start_date='2015-01-01',
    end_date='2018-02-07',
    market='CN_STOCK_A',
    instrument_list='600009.SHA',
    max_count=0
)


# Python 代码入口函数，input_1/2/3 对应三个输入端，data_1/2/3 对应三个输出端
def m24_run_bigquant_run(input_1, input_2, input_3):
    fields = ['open', 'high', 'low', 'close', 'volume']
    input_1_df = input_1.read_pickle()
    ins = input_1_df['instruments']
    start_date = input_1_df['start_date']
    end_date = input_1_df['end_date']
    df = D.history_data(ins, start_date, end_date, fields)
    data_1 = DataSource.write_df(df)
    return Outputs(data_1=data_1, data_2=None, data_3=None)


m24 = M.cached.v3(
    input_1=m25.data,
    run=m24_run_bigquant_run
)


# Python 代码入口函数，input_1/2/3 对应三个输入端，data_1/2/3 对应三个输出端
def m16_run_bigquant_run(input_1, input_2, input_3):
    input_ds = input_1
    df = input_ds.read_df()
    df['return'] = (df.close.shift(-10) / df.close - 1)
    df['label'] = np.where(df['return'] > 0, 1, 0)
    ds = DataSource.write_df(df)
    return Outputs(data_1=ds)


m16 = M.cached.v3(
    input_1=m24.data_1,
    run=m16_run_bigquant_run
)

m1 = M.derived_feature_extractor.v2(
    input_data=m24.data_1,
    features=m8.data,
    date_col='date',
    instrument_col='instrument',
    user_functions={}
)

m18 = M.join.v3(
    data1=m16.data_1,
    data2=m1.data,
    on='date',
    how='inner',
    sort=True
)

m19 = M.dropnan.v1(
    input_data=m18.data
)

m20 = M.filter.v3(
    input_data=m19.data,
    expr='date<\'2017-03-01\'',
    output_left_data=False
)

m22 = M.dl_convert_to_bin.v1(
    input_data=m20.data,
    features=m8.data,
    window_size=50
)

m6 = M.dl_model_train.v1(
    input_model=m5.data,
    training_data=m22.data,
    optimizer='Adam',
    loss='binary_crossentropy',
    metrics='accuracy',
    batch_size=2048,
    epochs=10,
    n_gpus=1,
    verbose='1:输出进度条记录'
)

m21 = M.filter.v3(
    input_data=m19.data,
    expr='date>\'2017-03-01\'',
    output_left_data=False
)

m23 = M.dl_convert_to_bin.v1(
    input_data=m21.data,
    features=m8.data,
    window_size=50
)

m7 = M.dl_model_predict.v1(
    trained_model=m6.data,
    input_data=m23.data,
    batch_size=10240,
    n_gpus=2,
    verbose='2:每个epoch输出一行记录'
)


# Python 代码入口函数，input_1/2/3 对应三个输入端，data_1/2/3 对应三个输出端
def m2_run_bigquant_run(input_1, input_2, input_3):
    input_series = input_1
    input_df = input_2
    test_data = input_df.read_pickle()
    pred_label = input_series.read_pickle()

    pred_result = pred_label.reshape(pred_label.shape[0])
    dt = input_3.read_df()['date'][-1 * len(pred_result):]
    pred_df = pd.Series(pred_result, index=dt)
    ds = DataSource.write_df(pred_df)

    pred_label = np.where(pred_label > 0.5, 1, 0)
    labels = test_data['y']
    print('准确率%s' % (np.mean(pred_label == labels)))

    return Outputs(data_1=ds)


m2 = M.cached.v3(
    input_1=m7.data,
    input_2=m23.data,
    input_3=m21.data,
    run=m2_run_bigquant_run
)


# 回测引擎：每日数据处理函数，每天执行一次
def m26_handle_data_bigquant_run(context, data):
    # 按日期过滤得到今日的预测数据
    try:
        prediction = context.prediction[data.current_dt.strftime('%Y-%m-%d')]
    except KeyError as e:
        return

    instrument = context.instruments[0]
    sid = context.symbol(instrument)
    cur_position = context.portfolio.positions[sid].amount
    # print('date: ',data.current_dt, '持仓： ', cur_position)

    # 交易逻辑
    if prediction > 0.5 and cur_position == 0:
        context.order_target_percent(context.symbol(instrument), 1)
        print(data.current_dt, '买入！')

    elif prediction < 0.5 and cur_position > 0:
        context.order_target_percent(context.symbol(instrument), 0)
        print(data.current_dt, '卖出！')


# 回测引擎：准备数据，只执行一次
def m26_prepare_bigquant_run(context):
    pass


# 回测引擎：初始化函数，只执行一次
def m26_initialize_bigquant_run(context):
    # 加载预测数据
    context.prediction = context.options['data'].read_df()

    # 系统已经设置了默认的交易手续费和滑点，要修改手续费可使用如下函数
    context.set_commission(PerOrder(buy_cost=0.0003, sell_cost=0.0013, min_cost=5))


# 回测引擎：每个单位时间开始前调用一次，即每日开盘前调用一次。
def m26_before_trading_start_bigquant_run(context, data):
    pass


m26 = M.trade.v3(
    instruments=m25.data,
    options_data=m2.data_1,
    start_date='2017-04-01',
    end_date='',
    handle_data=m26_handle_data_bigquant_run,
    prepare=m26_prepare_bigquant_run,
    initialize=m26_initialize_bigquant_run,
    before_trading_start=m26_before_trading_start_bigquant_run,
    volume_limit=0.025,
    order_price_field_buy='open',
    order_price_field_sell='close',
    capital_base=1000000,
    benchmark='000300.SHA',
    auto_cancel_non_tradable_orders=True,
    data_frequency='daily',
    price_type='真实价格',
    plot_charts=True,
    backtest_only=False,
    amount_integer=False
)