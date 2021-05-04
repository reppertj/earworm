/**
 *
 * ConditionalWrapper
 *
 */

export function ConditionalWrapper({
  condition,
  wrapper,
  children,
}: {
  condition: boolean;
  wrapper: (
    children: React.ReactElement<any, any>,
  ) => React.ReactElement<any, any>;
  children: React.ReactElement<any, any>;
}): React.ReactElement<any, any> {
  return condition ? wrapper(children) : children;
}
