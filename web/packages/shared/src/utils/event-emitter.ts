import { cloneDeep } from 'lodash-es';

export interface ISubscribe {
    topic: string;
    attrs?: Record<string, any>;
    callbacks: ((...args: any[]) => void)[];
}

/**
 * Publish&Subscribe implementation class
 */
export class EventEmitter<T extends ISubscribe = ISubscribe> {
    private subscribeHandles: T[];
    constructor() {
        this.subscribeHandles = [];
    }

    /**
     * Publish
     * @param {string} topic Topic
     * @param {...any[]} args The arguments to execute the callback
     */
    publish(topic: T['topic'], ...args: Parameters<T['callbacks'][number]>): void {
        const subscriber = this.subscribeHandles.find(
            (subscriber: T) => subscriber.topic === topic,
        );
        if (!subscriber) return;
        subscriber.callbacks.forEach(cb => cb(...args));
    }

    /**
     * Subscribe
     * @param {string} topic Topic
     * @param {Function} callback Callback function
     * @param {Object} attrs Attributes
     * @returns {boolean} Whether subscribed the topic before
     */
    subscribe(topic: T['topic'], callback: T['callbacks'][number], attrs?: T['attrs']): boolean {
        const subscriber = this.subscribeHandles.find(
            (subscriber: T) => subscriber.topic === topic,
        );
        if (subscriber) {
            attrs && (subscriber.attrs = attrs);
            subscriber.callbacks.push(callback);
            return true;
        }

        this.subscribeHandles.push({
            topic,
            attrs,
            callbacks: [callback],
        } as T);
        return false;
    }

    /**
     * Unsubscribe
     * @param {string} topic Topic
     * @param {Function} callback callback to be cleared, if not passed, clear all the subscriptions of the theme
     * @returns {boolean} Whether the topic has been cleared
     */
    unsubscribe(topic: T['topic'], callback?: T['callbacks'][number]): boolean {
        if (!callback) {
            this.subscribeHandles = this.subscribeHandles.filter(
                (subscribers: T) => subscribers.topic !== topic,
            );
            return true;
        }

        let isEmpty = false;
        this.subscribeHandles = this.subscribeHandles.reduce((handles: T[], subscriber: T) => {
            if (subscriber.topic === topic) {
                subscriber.callbacks = subscriber.callbacks.filter(cb => cb !== callback);
                isEmpty = !subscriber.callbacks?.length;
                if (isEmpty) return handles;
            }

            return [...handles, subscriber];
        }, []);

        return isEmpty;
    }

    /**
     * Get the first subscriber based on the topic
     * @param {string} topic Topic
     * @returns {object}
     */
    getSubscriber(topic: T['topic']): Readonly<T> | undefined {
        const subscriber = this.subscribeHandles.find(
            (subscriber: T) => subscriber.topic === topic,
        );
        if (!subscriber) return;

        // Deep copy to prevent the original data from being tampered with
        return cloneDeep(subscriber);
    }

    /**
     * Get all topics
     * @returns {string[]}
     */
    getTopics(): T['topic'][] {
        return this.subscribeHandles.map((subscriber: T) => subscriber.topic);
    }

    /**
     * Destroy all subscriptions
     */
    destroy(): void {
        this.subscribeHandles = [];
    }
}

export default new EventEmitter();
