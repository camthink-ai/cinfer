/* eslint-disable no-console */
import mqtt from 'mqtt';
import { throttle } from 'lodash-es';
import { safeJsonParse } from '@milesight/shared/src/utils/tools';
import { EventEmitter } from '@milesight/shared/src/utils/event-emitter';
import { splitExchangeTopic, batchPush } from './helper';
import { EVENT_TYPE, THROTTLE_TIME, BATCH_PUSH_TIME } from './constant';
import type { IEventEmitter, MqttMessageData, CallbackType } from './types';

interface MqttOptions extends mqtt.IClientOptions {
    url: string;
    debug?: boolean;
}

type MqttMessageDataType = MqttMessageData | undefined;

const DEFAULT_OPTIONS: mqtt.IClientOptions = {
    /** Clean messages while offline */
    clean: true,
    /** Interval between two reconnection */
    reconnectPeriod: 5000,
    /** Time to wait before a CONNACK is received */
    connectTimeout: 6000,
    /**
     * If connection is broken and reconnects, subscribed topics are automatically subscribed again */
    resubscribe: true,
    /**
     * Whether to verify the server certificate chain and address name
     */
    rejectUnauthorized: false,
    /** WebSocket connection options */
    wsOptions: {},
    /** Heartbeat interval to keep alive */
    keepalive: 50,
    /** Whether to reschedule ping when the network is disconnected */
    reschedulePings: true,
};

/**
 * MQTT service class
 */
class MqttService {
    private debug: boolean = false;
    private client: mqtt.MqttClient | null = null;
    private upTopic: string = '';
    private downTopic: string = '';
    private readonly eventEmitter: EventEmitter<IEventEmitter> = new EventEmitter(); // Event bus

    status: 'connecting' | 'connected' | 'disconnected' = 'disconnected';

    constructor({ url, debug, ...options }: MqttOptions) {
        if (!options.username || !options.password) {
            throw new Error('MQTT username or password is required');
        }

        this.debug = debug || false;
        this.status = 'connecting';
        this.client = mqtt.connect(url, { ...DEFAULT_OPTIONS, ...options });
        this.upTopic = `beaver-iot/${options.username}/uplink`;
        this.downTopic = `beaver-iot/${options.username}/downlink`;
        this.init();
    }

    init() {
        if (!this.client) return;

        this.client.on('connect', () => {
            const topic = this.downTopic;

            this.status = 'connected';
            this.client?.subscribe(topic, (err, granted) => {
                if (err) {
                    this.log([`MQTT subscribe ${topic} failed:`, err], 'error');
                } else {
                    this.log([`MQTT subscribe ${topic} success:`, granted]);
                }
            });
        });

        this.client.on('reconnect', () => {
            this.status = 'connecting';
            this.log('MQTT reconnecting...');
        });

        this.client.on('disconnect', () => {
            this.status = 'disconnected';
            this.log('disconnected');
        });

        this.client.on('offline', () => {
            this.status = 'disconnected';
            this.log('MQTT offline');
        });

        this.client.on('error', err => {
            this.status = 'disconnected';
            this.client?.end();
            this.log(['MQTT error:', err]);
        });

        // WS messages are pushed in batches
        type Queue = { topics: string[]; data: any[] };
        const { run: runWsPush, cancel: cancelWsPush } = batchPush((queue: Queue[][]) => {
            const batchPushQueue = queue.reduce(
                (bucket, item) => {
                    const [{ topics, data }] = item || {};

                    (topics || []).forEach(topic => {
                        if (bucket[topic]) {
                            bucket[topic].push(data);
                        } else {
                            bucket[topic] = [data];
                        }
                    });

                    return bucket;
                },
                {} as Record<string, any[]>,
            );

            Object.keys(batchPushQueue).forEach(topic => {
                const data = batchPushQueue[topic];
                this.eventEmitter.publish(topic, data);
            });
        }, BATCH_PUSH_TIME);

        this.client.on('message', (topic, message) => {
            this.log([`MQTT message received: topic=${topic}, message=${message.toString()}`]);
            if (this.downTopic !== topic) return;

            const { event_type: eventType, payload } =
                (safeJsonParse(message.toString()) as MqttMessageDataType) || {};

            switch (eventType) {
                case EVENT_TYPE.EXCHANGE: {
                    const { entity_key: topics } = payload || {};
                    runWsPush({
                        topics: topics?.map(topic => `${eventType}:${topic}`),
                        data: payload,
                    });
                    break;
                }
                // case 'Heartbeat':
                //     this.log([`MQTT Heartbeat message received:`, payload]);
                //     break;
                default:
                    break;
            }
        });
    }

    log(message: any | any[], level?: 'info' | 'warn' | 'error') {
        if (!this.debug) return;

        const logMessage = Array.isArray(message) ? message : [message];
        switch (level) {
            case 'info':
                console.info(...logMessage);
                break;
            case 'warn':
                console.warn(...logMessage);
                break;
            case 'error':
                console.error(...logMessage);
                break;
            default:
                console.log(...logMessage);
                break;
        }
    }

    private async emit() {
        if (this.status !== 'connected') return;

        const topics = this.eventEmitter.getTopics();
        // Extract the 'Exchange' type from the topic
        const topicPayload = splitExchangeTopic(topics);

        Object.entries(topicPayload).forEach(([type, data]) => {
            switch (type) {
                case EVENT_TYPE.EXCHANGE: {
                    const message: MqttMessageData = {
                        event_type: type,
                        payload: {
                            entity_key: data,
                        },
                    };
                    this.client?.publish(this.upTopic, JSON.stringify(message));
                    break;
                }
                default:
                    break;
            }
        });
    }

    /**
     * Subscribe to topics
     * @param {string | string[]} topics - Subscribed topics (supports passing in individual topics or lists of topics)
     * @param {Function} cb - Subscription callback
     * @returns After a successful subscription, a function is returned to cancel the subscription
     */
    subscribe(topics: string | string[], cb: CallbackType) {
        const _topics = Array.isArray(topics) ? topics : [topics];
        const publish = throttle(this.emit.bind(this), THROTTLE_TIME);

        _topics.forEach(topic => {
            // Whether you have subscribed
            const isSubscribed = this.eventEmitter.subscribe(topic, cb);
            if (!isSubscribed) {
                publish();
            }
        });
        return () => {
            this.unsubscribe(_topics, cb);
        };
    }

    /**
     * unsubscribe
     * @param {string | string[]} topics - Subscribed topics (supports passing in individual topics or lists of topics)
     * @param {Function} cb - Subscription callback
     */
    unsubscribe(topics: string | string[], cb?: CallbackType) {
        const _topics = Array.isArray(topics) ? topics : [topics];

        _topics.forEach(topic => {
            const isEmpty = this.eventEmitter.unsubscribe(topic, cb);

            isEmpty && this.emit();
        });
    }
}

export default MqttService;
